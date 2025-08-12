import jwt
import json
import os
import traceback
from datetime import datetime, timezone, timedelta
from functools import wraps

import bcrypt
import psycopg2
import psycopg2.extras
from flask import (
    Flask,
    jsonify,
    request,
    send_from_directory,
)
from psycopg2.pool import ThreadedConnectionPool
from sentence_transformers import SentenceTransformer

# ===============================
# КОНФИГУРАЦИЯ
# ===============================
# Пулы соединений с БД
db_manager = None

# JWT
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_EXPIRATION_HOURS = 24

# Модель для эмбеддингов
EMBEDDING_MODEL_NAME = "ai-forever/sbert_large_mt_nlu_ru"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# ===============================
# КЛАСС УПРАВЛЕНИЯ БД
# ===============================
class DatabaseManager:
    def __init__(self):
        self.pool = None

    def init_pool(self):
        try:
            self.pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=20,
                host=os.getenv("DB_HOST", "localhost"),
                port=os.getenv("DB_PORT", "5432"),
                database=os.getenv("DB_NAME", "resource_map"),
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", "1234"),
            )
            print("✅ Пул соединений с PostgreSQL инициализирован")
        except Exception as e:
            print(f"❌ Ошибка инициализации пула БД: {e}")
            raise

    def get_connection(self):
        if self.pool is None:
            self.init_pool()
        return self.pool.getconn()

    def put_connection(self, conn):
        if self.pool is not None:
            self.pool.putconn(conn)


db_manager = DatabaseManager()


# ===============================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ===============================
def generate_embedding(text: str):
    """Генерация эмбеддинга для текста"""
    try:
        embedding = embedding_model.encode(text)
        return embedding.tolist() if hasattr(embedding, "tolist") else embedding
    except Exception as e:
        print(f"Ошибка генерации эмбеддинга: {e}")
        return None


def generate_embedding_text(
    name="", description="", services="", address="", tags=None, main_service=""
):
    """Формирование текста для генерации эмбеддинга"""
    tags_str = ", ".join(tags) if tags else ""
    return f"{name} {main_service} {description} {services} {address} {tags_str}"


# ===============================
# JWT АУТЕНТИФИКАЦИЯ
# ===============================


def generate_jwt_token(user_id: int, email: str, role: str) -> str:
    """Генерация JWT токена для администратора"""
    payload = {
        "user_id": user_id,
        "email": email,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")


def verify_jwt_token(token: str):
    """Верификация JWT токена"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# Администраторы - Auth
def require_auth(f):
    """Декоратор для защиты эндпоинтов администратора"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token or not token.startswith("Bearer "):
            return jsonify({"error": "Токен не предоставлен"}), 401
        token = token.split(" ")[1]
        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({"error": "Неверный или истекший токен"}), 401
        request.current_admin = payload
        return f(*args, **kwargs)

    return decorated_function


def generate_owner_jwt_token(owner_id: int, email: str) -> str:
    """Генерация JWT токена для владельца"""
    payload = {
        "owner_id": owner_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm="HS256")


def require_owner_auth(f):
    """Декоратор для защиты эндпоинтов владельца"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token or not token.startswith("Bearer "):
            return jsonify({"error": "Токен не предоставлен"}), 401
        token = token.split(" ")[1]
        payload = verify_jwt_token(token)
        if not payload or "owner_id" not in payload:
            return jsonify({"error": "Неверный или истекший токен"}), 401
        request.current_owner = payload
        return f(*args, **kwargs)

    return decorated_function


# ===============================
# API ЭНДПОИНТЫ
# ===============================
app = Flask(__name__, static_folder="static")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/admin")
def admin_index():
    return send_from_directory("static", "admin_index.html")


@app.route("/owner")
def owner_index():
    return send_from_directory("static", "owner_index.html")


@app.route("/api/search", methods=["GET"])
def semantic_search():
    """Семантический поиск организаций"""
    try:
        query = request.args.get("q", "").strip()
        if not query:
            return jsonify({"organizations": []})

        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Генерация эмбеддинга для запроса
            query_embedding = generate_embedding(query)
            if not query_embedding:
                return jsonify({"organizations": []})

            # Поиск по косинусному сходству
            cursor.execute(
                """
                SELECT 
                    o.id, o.name, o.main_service, o.description, o.address,
                    o.phone, o.email, o.website, o.services, o.tags,
                    o.contact_person_name, o.contact_person_phone,
                    o.contact_person_email, o.contact_person_photo_url,
                    c.name as category_name,
                    1 - (o.embedding <=> %s::vector) as similarity
                FROM organizations o
                JOIN categories c ON o.category_id = c.id
                WHERE 1 - (o.embedding <=> %s::vector) > 0.25
                ORDER BY similarity DESC
                LIMIT 50
                """,
                (query_embedding, query_embedding),
            )
            results = cursor.fetchall()

            # Преобразуем результат
            organizations = []
            for row in results:
                org = dict(row)
                org["similarity"] = float(org["similarity"])
                organizations.append(org)

            return jsonify({"organizations": organizations})
    except Exception as e:
        print(f"Ошибка поиска: {e}")
        return jsonify({"organizations": []})


@app.route("/api/categories", methods=["GET"])
def get_categories():
    """Получение всех категорий"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("SELECT id, name FROM categories ORDER BY name")
            categories = cursor.fetchall()
            return jsonify([dict(cat) for cat in categories])
    except Exception as e:
        print(f"Ошибка получения категорий: {e}")
        return jsonify([]), 500


@app.route("/api/admin/categories", methods=["GET"])
@require_auth
def get_admin_categories():
    """Получение категорий для администратора с количеством организаций"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT c.id, c.name, COUNT(o.id) as organization_count
                FROM categories c
                LEFT JOIN organizations o ON c.id = o.category_id
                GROUP BY c.id, c.name
                ORDER BY c.name
            """)
            categories = cursor.fetchall()
            return jsonify({
                "categories": [dict(cat) for cat in categories]
            })
    except Exception as e:
        print(f"Ошибка получения категорий для админа: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


@app.route("/api/organizations", methods=["GET"])
def get_organizations():
    """Получение всех организаций"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT 
                    o.id, o.name, o.main_service, o.description, o.address,
                    o.phone, o.email, o.website, o.services, o.tags,
                    o.contact_person_name, o.contact_person_phone,
                    o.contact_person_email, o.contact_person_photo_url,
                    c.name as category_name
                FROM organizations o
                JOIN categories c ON o.category_id = c.id
                ORDER BY o.name
                """
            )
            organizations = cursor.fetchall()
            return jsonify({"organizations": [dict(org) for org in organizations]})
    except Exception as e:
        print(f"Ошибка получения организаций: {e}")
        return jsonify({"organizations": []}), 500


@app.route("/api/organizations/<int:org_id>", methods=["GET"])
def get_organization(org_id):
    """Получение одной организации"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT 
                    o.id, o.name, o.main_service, o.description, o.address,
                    o.phone, o.email, o.website, o.services, o.tags,
                    o.contact_person_name, o.contact_person_role,
                    o.contact_person_phone, o.contact_person_email,
                    o.contact_person_photo_url, o.created_at, o.updated_at,
                    c.name as category_name, c.id as category_id
                FROM organizations o
                JOIN categories c ON o.category_id = c.id
                    WHERE o.id = %s
                """,
                (org_id,),
            )
            org = cursor.fetchone()
            if not org:
                return jsonify({"error": "Организация не найдена"}), 404
            return jsonify(dict(org))
    except Exception as e:
        print(f"Ошибка получения организации: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


@app.route("/api/admin/organizations/<int:org_id>", methods=["GET"])
@require_auth
def get_admin_organization(org_id):
    """Получение одной организации для администратора"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT 
                    o.id, o.name, o.main_service, o.description, o.address,
                    o.phone, o.email, o.website, o.services, o.tags,
                    o.contact_person_name, o.contact_person_role,
                    o.contact_person_phone, o.contact_person_email,
                    o.contact_person_photo_url, o.created_at, o.updated_at,
                    c.name as category_name, c.id as category_id
                FROM organizations o
                JOIN categories c ON o.category_id = c.id
                WHERE o.id = %s
                """,
                (org_id,),
            )
            org = cursor.fetchone()
            if not org:
                return jsonify({"error": "Организация не найдена"}), 404
            
            org_dict = dict(org)
            org_dict["tags"] = org["tags"] if org["tags"] else []
            org_dict["is_active"] = True  # Все организации активны по умолчанию
            
            return jsonify({"organization": org_dict})
    except Exception as e:
        print(f"Ошибка получения организации для админа: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


# ===============================
# АДМИН: АВТОРИЗАЦИЯ
# ===============================
@app.route("/api/admin/login", methods=["POST"])
def admin_login():
    """Вход администратора"""
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"error": "Email и пароль обязательны"}), 400

        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                "SELECT id, email, password_hash FROM admins WHERE email = %s",
                (email.lower(),),
            )
            admin = cursor.fetchone()

            if not admin:
                return jsonify({"error": "Неверные учетные данные"}), 401

            if not bcrypt.checkpw(
                password.encode("utf-8"), admin["password_hash"].encode("utf-8")
            ):
                return jsonify({"error": "Неверные учетные данные"}), 401

            token = generate_jwt_token(admin["id"], admin["email"], "admin")
            return jsonify(
                {
                    "token": token,
                    "admin": {
                        "id": admin["id"],
                        "email": admin["email"],
                    },
                }
            )
    except Exception as e:
        print(f"Ошибка входа администратора: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


@app.route("/api/admin/logout", methods=["POST"])
@require_auth
def admin_logout():
    """Выход администратора (на клиенте удаляется токен)"""
    return jsonify({"message": "Выход выполнен"})


@app.route("/api/admin/profile", methods=["GET"])
@require_auth
def admin_profile():
    """Получение профиля администратора"""
    return jsonify(
        {
            "id": request.current_admin["user_id"],
            "email": request.current_admin["email"],
        }
    )


@app.route("/api/admin/stats", methods=["GET"])
@require_auth
def admin_stats():
    """Получение статистики для дашборда"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Общая статистика
            cursor.execute("SELECT COUNT(*) as total FROM organizations")
            total_orgs = cursor.fetchone()["total"]
            
            cursor.execute("SELECT COUNT(*) as total FROM categories")
            total_cats = cursor.fetchone()["total"]
            
            # Последние организации
            cursor.execute("""
                SELECT o.name, c.name as category, o.created_at
                FROM organizations o
                JOIN categories c ON o.category_id = c.id
                ORDER BY o.created_at DESC
                LIMIT 5
            """)
            recent_orgs = cursor.fetchall()
            
            return jsonify({
                "stats": {
                    "total_organizations": total_orgs,
                    "active_organizations": total_orgs,  # Все организации активны по умолчанию
                    "inactive_organizations": 0,
                    "total_categories": total_cats
                },
                "recent_organizations": [
                    {
                        "name": org["name"],
                        "category": org["category"],
                        "created_at": org["created_at"].isoformat() if org["created_at"] else None
                    }
                    for org in recent_orgs
                ]
            })
    except Exception as e:
        print(f"Ошибка получения статистики: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


# ===============================
# АДМИН: ОРГАНИЗАЦИИ
# ===============================
@app.route("/api/admin/organizations", methods=["GET"])
@require_auth
def get_admin_organizations():
    """Получение организаций для администратора с пагинацией"""
    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 20))
        search = request.args.get("search", "").strip()
        category_id = request.args.get("category_id", "")
        status = request.args.get("status", "all")
        
        offset = (page - 1) * limit
        
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Базовый запрос
            base_query = """
                FROM organizations o
                JOIN categories c ON o.category_id = c.id
                WHERE 1=1
            """
            params = []
            
            # Поиск по названию
            if search:
                base_query += " AND o.name ILIKE %s"
                params.append(f"%{search}%")
            
            # Фильтр по категории
            if category_id:
                base_query += " AND o.category_id = %s"
                params.append(category_id)
            
            # Подсчет общего количества
            count_query = f"SELECT COUNT(*) as total {base_query}"
            cursor.execute(count_query, params)
            total = cursor.fetchone()["total"]
            
            # Получение организаций
            orgs_query = f"""
                SELECT 
                    o.id, o.name, o.main_service, o.description, o.address,
                    o.phone, o.email, o.website, o.services, o.tags,
                    o.contact_person_name, o.contact_person_role,
                    o.contact_person_phone, o.contact_person_email,
                    o.contact_person_photo_url, o.created_at, o.updated_at,
                    c.name as category, c.id as category_id
                {base_query}
                ORDER BY o.name
                LIMIT %s OFFSET %s
            """
            cursor.execute(orgs_query, params + [limit, offset])
            organizations = cursor.fetchall()
            
            # Преобразование в словари
            orgs_list = []
            for org in organizations:
                org_dict = dict(org)
                org_dict["tags"] = org["tags"] if org["tags"] else []
                orgs_list.append(org_dict)
            
            pages = (total + limit - 1) // limit
            
            return jsonify({
                "organizations": orgs_list,
                "total": total,
                "pages": pages,
                "current_page": page
            })
    except Exception as e:
        print(f"Ошибка получения организаций: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


@app.route("/api/admin/organizations", methods=["POST"])
@require_auth
def create_organization():
    """Создание новой организации"""
    try:
        data = request.get_json()

        required_fields = ["name", "category_id", "description", "address"]
        for field in required_fields:
            if not data.get(field):
                return jsonify({"error": f"Поле {field} обязательно"}), 400

        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            # Проверка существования категории
            cursor.execute(
                "SELECT id FROM categories WHERE id = %s", (data["category_id"],)
            )
            if not cursor.fetchone():
                return jsonify({"error": "Категория не существует"}), 400

            # Генерация эмбеддинга
            text_for_embedding = generate_embedding_text(
                name=data["name"],
                description=data["description"],
                services=data.get("services", ""),
                address=data.get("address", ""),
                tags=data.get("tags", []),
                main_service=data.get("main_service", ""),
            )
            embedding = generate_embedding(text_for_embedding)

            cursor.execute(
                """
                INSERT INTO organizations (
                    name, main_service, category_id, description, address,
                    phone, email, website, services, tags,
                    contact_person_name, contact_person_role,
                    contact_person_phone, contact_person_email,
                    contact_person_photo_url, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    data["name"],
                    data.get("main_service"),
                    data["category_id"],
                    data["description"],
                    data["address"],
                    data.get("phone"),
                    data.get("email"),
                    data.get("website"),
                    data.get("services"),
                    data.get("tags"),
                    data.get("contact_person_name"),
                    data.get("contact_person_role"),
                    data.get("contact_person_phone"),
                    data.get("contact_person_email"),
                    data.get("contact_person_photo_url"),
                    embedding,
                ),
            )
            org_id = cursor.fetchone()[0]
            conn.commit()

            return jsonify({"id": org_id}), 201
    except Exception as e:
        print(f"Ошибка создания организации: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


@app.route("/api/admin/organizations/<int:org_id>", methods=["PUT"])
@require_auth
def update_organization(org_id):
    """Обновление организации"""
    try:
        data = request.get_json()

        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            # Проверка существования категории
            if "category_id" in data and data["category_id"]:
                cursor.execute(
                    "SELECT id FROM categories WHERE id = %s", (data["category_id"],)
                )
                if not cursor.fetchone():
                    return jsonify({"error": "Категория не существует"}), 400

            # Генерация эмбеддинга
            text_for_embedding = generate_embedding_text(
                name=data.get("name", ""),
                description=data.get("description", ""),
                services=data.get("services", ""),
                address=data.get("address", ""),
                tags=data.get("tags", []),
                main_service=data.get("main_service", ""),
            )
            embedding = generate_embedding(text_for_embedding)

            cursor.execute(
                """
                UPDATE organizations SET
                    name = %s, main_service = %s, category_id = %s,
                    description = %s, address = %s, phone = %s,
                    email = %s, website = %s, services = %s, tags = %s,
                    contact_person_name = %s, contact_person_role = %s,
                    contact_person_phone = %s, contact_person_email = %s,
                    contact_person_photo_url = %s, embedding = %s,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (
                    data.get("name"),
                    data.get("main_service"),
                    data.get("category_id"),
                    data.get("description"),
                    data.get("address"),
                    data.get("phone"),
                    data.get("email"),
                    data.get("website"),
                    data.get("services"),
                    data.get("tags"),
                    data.get("contact_person_name"),
                    data.get("contact_person_role"),
                    data.get("contact_person_phone"),
                    data.get("contact_person_email"),
                    data.get("contact_person_photo_url"),
                    embedding,
                    org_id,
                ),
            )
            conn.commit()
            return jsonify({"message": "Организация обновлена"})
    except Exception as e:
        print(f"Ошибка обновления организации: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


@app.route("/api/admin/organizations/<int:org_id>", methods=["DELETE"])
@require_auth
def delete_organization(org_id: int):
    """Удаление организации"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM organizations WHERE id = %s", (org_id,))
            if cursor.rowcount == 0:
                return jsonify({"error": "Организация не найдена"}), 404
            conn.commit()
            return jsonify({"message": "Организация успешно удалена"})
    except Exception as e:
        print(f"Ошибка удаления организации: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


# ===============================
# АДМИН: МОДЕРАЦИЯ ЗАЯВОК
# ===============================
@app.route("/api/admin/pending-requests", methods=["GET"])
@require_auth
def get_pending_requests():
    """Получение всех заявок на модерацию"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT 
                    pr.id, pr.request_type, pr.data, pr.owner_email, pr.created_at,
                    pr.status, pr.reviewed_at, a.email as reviewed_by_email
                FROM pending_requests pr
                LEFT JOIN admins a ON pr.reviewed_by = a.id
                WHERE pr.status = 'pending'
                ORDER BY pr.created_at DESC
                """
            )
            requests = cursor.fetchall()
            return jsonify([dict(req) for req in requests])
    except Exception as e:
        print(f"Ошибка получения заявок: {e}")
        return jsonify([]), 500


@app.route("/api/admin/organization-add-requests", methods=["GET"])
@require_auth
def get_organization_add_requests():
    """Получение заявок на добавление организаций"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT 
                    oar.id, oar.organization_name, oar.description, oar.address,
                    oar.phone, oar.email, oar.website, oar.services, oar.tags,
                    oar.contact_person_name, oar.contact_person_role,
                    oar.contact_person_phone, oar.contact_person_email,
                    oar.contact_person_photo_url, oar.requester_name, oar.requester_email,
                    oar.requester_phone, oar.status, oar.created_at,
                    oar.reviewed_at, a.email as reviewed_by_email,
                    c.name as category_name
                FROM organization_add_requests oar
                LEFT JOIN categories c ON oar.category_id = c.id
                LEFT JOIN admins a ON oar.reviewed_by = a.id
                ORDER BY oar.created_at DESC
                """
            )
            requests = cursor.fetchall()
            return jsonify({"requests": [dict(req) for req in requests]})
    except Exception as e:
        print(f"Ошибка получения заявок на добавление организаций: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


@app.route("/api/admin/organization-add-requests/<int:request_id>/approve", methods=["POST"])
@require_auth
def approve_organization_add_request(request_id: int):
    """Одобрение заявки на добавление организации"""
    try:
        admin_id = request.current_admin["user_id"]
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM organization_add_requests WHERE id = %s AND status = 'pending'",
                (request_id,),
            )
            request_data = cursor.fetchone()
            if not request_data:
                return jsonify({"error": "Заявка не найдена или уже обработана"}), 404

            # Создание новой организации
            text_for_embedding = generate_embedding_text(
                name=request_data["organization_name"],
                description=request_data.get("description", ""),
                services=request_data.get("services", ""),
                address=request_data.get("address", ""),
                tags=request_data.get("tags", []),
                main_service="",
            )
            embedding = generate_embedding(text_for_embedding)

            cursor.execute(
                """
                INSERT INTO organizations (
                    name, main_service, category_id, description, address,
                    phone, email, website, services, tags,
                    contact_person_name, contact_person_role,
                    contact_person_phone, contact_person_email,
                    contact_person_photo_url, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    request_data["organization_name"],
                    "",
                    request_data["category_id"],
                    request_data["description"],
                    request_data.get("address"),
                    request_data.get("phone"),
                    request_data.get("email"),
                    request_data.get("website"),
                    request_data.get("services"),
                    request_data.get("tags"),
                    request_data.get("contact_person_name"),
                    request_data.get("contact_person_role"),
                    request_data.get("contact_person_phone"),
                    request_data.get("contact_person_email"),
                    request_data.get("contact_person_photo_url"),
                    embedding,
                ),
            )
            new_org_id = cursor.fetchone()[0]

            # Обновляем статус заявки
            cursor.execute(
                """
                UPDATE organization_add_requests
                SET status = 'approved', reviewed_by = %s, reviewed_at = NOW()
                WHERE id = %s
                """,
                (admin_id, request_id),
            )
            conn.commit()
            return jsonify({"message": "Заявка одобрена, организация создана"})
    except Exception as e:
        print(f"Ошибка одобрения заявки на добавление организации: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


@app.route("/api/admin/organization-add-requests/<int:request_id>/reject", methods=["POST"])
@require_auth
def reject_organization_add_request(request_id: int):
    """Отклонение заявки на добавление организации"""
    try:
        admin_id = request.current_admin["user_id"]
        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE organization_add_requests
                SET status = 'rejected', reviewed_by = %s, reviewed_at = NOW()
                WHERE id = %s AND status = 'pending'
                """,
                (admin_id, request_id),
            )
            if cursor.rowcount == 0:
                return jsonify({"error": "Заявка не найдена или уже обработана"}), 404
            conn.commit()
            return jsonify({"message": "Заявка отклонена"})
    except Exception as e:
        print(f"Ошибка отклонения заявки на добавление организации: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


@app.route("/api/admin/pending-requests/<int:request_id>/approve", methods=["POST"])
@require_auth
def approve_pending_request(request_id: int):
    """Одобрение заявки"""
    try:
        admin_id = request.current_admin["user_id"]
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                "SELECT id, request_type, data, owner_email FROM pending_requests WHERE id = %s AND status = 'pending'",
                (request_id,),
            )
            request_data = cursor.fetchone()
            if not request_data:
                return jsonify({"error": "Заявка не найдена или уже обработана"}), 404

            request_type = request_data["request_type"]
            data = request_data["data"]

            if request_type == "new_org":
                # Создание новой организации
                text_for_embedding = generate_embedding_text(
                    name=data["organization_name"],
                    description=data.get("description", ""),
                    services=data.get("services", ""),
                    address=data.get("address", ""),
                    tags=data.get("tags", []),
                    main_service=data.get("main_service", ""),
                )
                embedding = generate_embedding(text_for_embedding)

                cursor.execute(
                    """
                    INSERT INTO organizations (
                        name, main_service, category_id, description, address,
                        phone, email, website, services, tags,
                        contact_person_name, contact_person_role,
                        contact_person_phone, contact_person_email,
                        contact_person_photo_url, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        data["organization_name"],
                        data.get("main_service"),
                        data["category_id"],
                        data["description"],
                        data["address"],
                        data.get("phone"),
                        data.get("email"),
                        data.get("website"),
                        data.get("services"),
                        data.get("tags"),
                        data.get("contact_person_name"),
                        data.get("contact_person_role"),
                        data.get("contact_person_phone"),
                        data.get("contact_person_email"),
                        data.get("contact_person_photo_url"),
                        embedding,
                    ),
                )
                new_org_id = cursor.fetchone()[0]

                # Создание владельца
                password_hash = bcrypt.hashpw(
                    data["password"].encode("utf-8"), bcrypt.gensalt()
                ).decode("utf-8")
                cursor.execute(
                    """
                    INSERT INTO organization_owners 
                    (full_name, email, phone, password_hash, organization_id, is_verified, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        data["full_name"],
                        data["email"],
                        data.get("phone", ""),
                        password_hash,
                        new_org_id,
                        True,
                        True,
                    ),
                )

            elif request_type == "claim_org":
                # Привязка владельца к существующей организации
                cursor.execute(
                    "SELECT id FROM organizations WHERE id = %s", (data["org_id"],)
                )
                if not cursor.fetchone():
                    return jsonify({"error": "Организация не существует"}), 400

                # Проверка, не привязан ли уже владелец
                cursor.execute(
                    "SELECT id FROM organization_owners WHERE email = %s AND organization_id = %s",
                    (data["email"], data["org_id"]),
                )
                if cursor.fetchone():
                    return jsonify(
                        {"error": "Владелец уже привязан к этой организации"}
                    ), 400

                password_hash = bcrypt.hashpw(
                    data["password"].encode("utf-8"), bcrypt.gensalt()
                ).decode("utf-8")
                cursor.execute(
                    """
                    INSERT INTO organization_owners 
                    (full_name, email, phone, password_hash, organization_id, is_verified, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        data["full_name"],
                        data["email"],
                        data.get("phone", ""),
                        password_hash,
                        data["org_id"],
                        True,
                        True,
                    ),
                )

            elif request_type == "update_org":
                # Обновление организации
                org_id = data["organization_id"]
                cursor.execute("SELECT id FROM organizations WHERE id = %s", (org_id,))
                if not cursor.fetchone():
                    return jsonify({"error": "Организация не найдена"}), 400

                text_for_embedding = generate_embedding_text(
                    name=data.get("name", ""),
                    description=data.get("description", ""),
                    services=data.get("services", ""),
                    address=data.get("address", ""),
                    tags=data.get("tags", []),
                    main_service=data.get("main_service", ""),
                )
                embedding = generate_embedding(text_for_embedding)

                cursor.execute(
                    """
                    UPDATE organizations SET
                        name = %s, main_service = %s, category_id = %s,
                        description = %s, address = %s, phone = %s,
                        email = %s, website = %s, services = %s, tags = %s,
                        contact_person_name = %s, contact_person_role = %s,
                        contact_person_phone = %s, contact_person_email = %s,
                        contact_person_photo_url = %s, embedding = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (
                        data.get("name"),
                        data.get("main_service"),
                        data.get("category_id"),
                        data.get("description"),
                        data.get("address"),
                        data.get("phone"),
                        data.get("email"),
                        data.get("website"),
                        data.get("services"),
                        data.get("tags"),
                        data.get("contact_person_name"),
                        data.get("contact_person_role"),
                        data.get("contact_person_phone"),
                        data.get("contact_person_email"),
                        data.get("contact_person_photo_url"),
                        embedding,
                        org_id,
                    ),
                )

            # Обновляем статус заявки
            cursor.execute(
                """
                UPDATE pending_requests
                SET status = 'approved', reviewed_by = %s, reviewed_at = NOW()
                WHERE id = %s
                """,
                (admin_id, request_id),
            )
            conn.commit()
            return jsonify({"message": "Заявка одобрена"})

    except Exception as e:
        print(f"Ошибка одобрения заявки: {e}")
        traceback.print_exc()
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


@app.route("/api/admin/pending-requests/<int:request_id>/reject", methods=["POST"])
@require_auth
def reject_pending_request(request_id: int):
    """Отклонение заявки"""
    try:
        admin_id = request.current_admin["user_id"]
        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE pending_requests
                SET status = 'rejected', reviewed_by = %s, reviewed_at = NOW()
                WHERE id = %s AND status = 'pending'
                """,
                (admin_id, request_id),
            )
            if cursor.rowcount == 0:
                return jsonify({"error": "Заявка не найдена или уже обработана"}), 404
            conn.commit()
            return jsonify({"message": "Заявка отклонена"})
    except Exception as e:
        print(f"Ошибка отклонения заявки: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


@app.route("/api/admin/owners/<int:owner_id>/deactivate", methods=["POST"])
@require_auth
def deactivate_owner(owner_id: int):
    """Деактивация владельца организации"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE organization_owners
                SET is_active = FALSE
                WHERE id = %s
                """,
                (owner_id,),
            )
            if cursor.rowcount == 0:
                return jsonify({"error": "Владелец не найден"}), 404
            conn.commit()
            return jsonify({"message": "Владелец деактивирован"})
    except Exception as e:
        print(f"Ошибка деактивации владельца: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


@app.route("/api/admin/owners", methods=["GET"])
@require_auth
def get_owners():
    """Получение списка владельцев организаций"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    oo.id, oo.full_name, oo.email, oo.phone, oo.is_verified, oo.is_active,
                    oo.created_at, o.name as organization_name
                FROM organization_owners oo
                LEFT JOIN organizations o ON oo.organization_id = o.id
                ORDER BY oo.created_at DESC
            """)
            owners = cursor.fetchall()
            return jsonify({"owners": [dict(owner) for owner in owners]})
    except Exception as e:
        print(f"Ошибка получения владельцев: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


# ===============================
# ВЛАДЕЛЕЦ: РЕГИСТРАЦИЯ И АВТОРИЗАЦИЯ
# ===============================
@app.route("/api/owner/register", methods=["POST"])
def register_owner():
    """Регистрация владельца (создание или привязка)"""
    try:
        data = request.get_json()

        required_fields = ["full_name", "email", "password", "organization_name"]
        for field in required_fields:
            if not data.get(field):
                return jsonify({"error": f"Поле {field} обязательно"}), 400

        if len(data["password"]) < 6:
            return jsonify({"error": "Пароль должен быть не менее 6 символов"}), 400

        email = data["email"].lower()
        if "@" not in email or "." not in email:
            return jsonify({"error": "Некорректный email адрес"}), 400

        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Проверка существования владельца
            cursor.execute(
                "SELECT id, is_verified FROM organization_owners WHERE email = %s",
                (email,),
            )
            existing_owner = cursor.fetchone()
            if existing_owner:
                if existing_owner["is_verified"]:
                    return jsonify(
                        {"error": "Владелец с таким email уже существует"}
                    ), 400
                # Можно обновить данные, если не подтверждён

            # Проверка существования организации
            cursor.execute(
                "SELECT id, name FROM organizations WHERE name = %s",
                (data["organization_name"],),
            )
            org_result = cursor.fetchone()

            request_type = "claim_org" if org_result else "new_org"
            pending_data = {
                "full_name": data["full_name"],
                "email": email,
                "phone": data.get("phone"),
                "password": data["password"],
                "organization_name": data["organization_name"],
            }

            if request_type == "claim_org":
                pending_data["org_id"] = org_result["id"]
                pending_data["category_id"] = data.get("category_id")
                pending_data["description"] = data.get("description")
                pending_data["address"] = data.get("address")
                pending_data["services"] = data.get("services")
                pending_data["tags"] = data.get("tags")
                pending_data["main_service"] = data.get("main_service")
                pending_data["contact_person_name"] = data.get("contact_person_name")
                pending_data["contact_person_phone"] = data.get("contact_person_phone")
                pending_data["contact_person_email"] = data.get("contact_person_email")
                pending_data["contact_person_photo_url"] = data.get(
                    "contact_person_photo_url"
                )

            # Создаём запрос на модерацию
            cursor.execute(
                """
                INSERT INTO pending_requests (request_type, data, owner_email)
                VALUES (%s, %s, %s)
                """,
                (request_type, psycopg2.extras.Json(pending_data), email),
            )
            conn.commit()

            return jsonify(
                {
                    "message": "Регистрация успешна. Запрос на создание организации отправлен на модерацию.",
                }
            ), 201
    except Exception as e:
        print(f"Ошибка регистрации владельца: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


@app.route("/api/owner/login", methods=["POST"])
def owner_login():
    """Вход владельца"""
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"error": "Email и пароль обязательны"}), 400

        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(
                "SELECT id, full_name, email, password_hash, organization_id, is_verified, is_active FROM organization_owners WHERE email = %s",
                (email.lower(),),
            )
            owner = cursor.fetchone()

            if not owner:
                return jsonify({"error": "Неверные учетные данные"}), 401

            if not owner["is_verified"]:
                return jsonify(
                    {"error": "Аккаунт ожидает подтверждения администратора"}
                ), 401

            if not owner["is_active"]:
                return jsonify({"error": "Аккаунт деактивирован"}), 401

            if not bcrypt.checkpw(password.encode("utf-8"), owner["password_hash"].encode("utf-8")):
                return jsonify({"error": "Неверные учетные данные"}), 401

            token = generate_owner_jwt_token(owner["id"], owner["email"])
            return jsonify(
                {
                    "token": token,
                    "owner": {
                        "id": owner["id"],
                        "full_name": owner["full_name"],
                        "email": owner["email"],
                        "organization_id": owner["organization_id"],
                    },
                }
            )
    except Exception as e:
        print(f"Ошибка авторизации владельца: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


@app.route("/api/owner/claim", methods=["POST"])
def claim_organization():
    """Привязка к существующей организации"""
    try:
        data = request.get_json()

        required_fields = [
            "full_name",
            "email",
            "password",
            "organization_name",
            "category_id",
        ]
        for field in required_fields:
            if not data.get(field):
                return jsonify({"error": f"Поле {field} обязательно"}), 400

        email = data["email"].lower()
        if "@" not in email or "." not in email:
            return jsonify({"error": "Некорректный email адрес"}), 400

        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            # Проверка существования организации
            cursor.execute(
                "SELECT id FROM organizations WHERE name = %s",
                (data["organization_name"],),
            )
            org = cursor.fetchone()
            if not org:
                return jsonify({"error": "Организация не найдена"}), 400

            # Проверка, не привязан ли уже владелец
            cursor.execute(
                "SELECT id FROM organization_owners WHERE email = %s", (email,)
            )
            if cursor.fetchone():
                return jsonify({"error": "Владелец с таким email уже существует"}), 400

            # Создаём заявку на модерацию
            pending_data = {
                "full_name": data["full_name"],
                "email": email,
                "phone": data.get("phone"),
                "password": data["password"],
                "org_id": org["id"],
                "organization_name": data["organization_name"],
                "category_id": data["category_id"],
                "description": data.get("description"),
                "address": data.get("address"),
                "services": data.get("services"),
                "tags": data.get("tags"),
                "main_service": data.get("main_service"),
                "contact_person_name": data.get("contact_person_name"),
                "contact_person_phone": data.get("contact_person_phone"),
                "contact_person_email": data.get("contact_person_email"),
                "contact_person_photo_url": data.get("contact_person_photo_url"),
            }

            cursor.execute(
                """
                INSERT INTO pending_requests (request_type, data, owner_email)
                VALUES ('claim_org', %s, %s)
                """,
                (psycopg2.extras.Json(pending_data), email),
            )
            conn.commit()

            return jsonify(
                {"message": "Запрос на привязку отправлен на модерацию"}
            ), 201
    except Exception as e:
        print(f"Ошибка привязки к организации: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


@app.route("/api/organizations/add-request", methods=["POST"])
def create_organization_add_request():
    """Создание заявки на добавление организации"""
    try:
        data = request.get_json()

        required_fields = ["organization_name", "category_id", "description", "requester_name", "requester_email"]
        for field in required_fields:
            if not data.get(field):
                return jsonify({"error": f"Поле {field} обязательно"}), 400

        email = data["requester_email"].lower()
        if "@" not in email or "." not in email:
            return jsonify({"error": "Некорректный email адрес"}), 400

        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            # Проверка существования организации
            cursor.execute(
                "SELECT id FROM organizations WHERE name = %s",
                (data["organization_name"],),
            )
            if cursor.fetchone():
                return jsonify({"error": "Организация с таким названием уже существует"}), 400

            # Создаём заявку на добавление
            cursor.execute(
                """
                INSERT INTO organization_add_requests (
                    organization_name, category_id, description, address, phone, email,
                    website, services, tags, contact_person_name, contact_person_role,
                    contact_person_phone, contact_person_email, contact_person_photo_url,
                    requester_name, requester_email, requester_phone
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    data["organization_name"],
                    data["category_id"],
                    data["description"],
                    data.get("address"),
                    data.get("phone"),
                    data.get("email"),
                    data.get("website"),
                    data.get("services"),
                    data.get("tags"),
                    data.get("contact_person_name"),
                    data.get("contact_person_role"),
                    data.get("contact_person_phone"),
                    data.get("contact_person_email"),
                    data.get("contact_person_photo_url"),
                    data["requester_name"],
                    email,
                    data.get("requester_phone"),
                ),
            )
            conn.commit()

            return jsonify(
                {"message": "Заявка на добавление организации отправлена на модерацию"}
            ), 201
    except Exception as e:
        print(f"Ошибка создания заявки на добавление организации: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


@app.route("/api/owner/organizations", methods=["PUT"])
@require_owner_auth
def update_owner_organization():
    """Обновление данных организации владельцем"""
    try:
        data = request.get_json()
        owner_id = request.current_owner["owner_id"]

        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            # Проверка, что владелец привязан к организации
            cursor.execute(
                """
                SELECT o.id FROM organizations o
                JOIN organization_owners oo ON o.id = oo.organization_id
                WHERE oo.id = %s
                """,
                (owner_id,),
            )
            org = cursor.fetchone()
            if not org:
                return jsonify({"error": "Организация не найдена"}), 404

            # Создаём запрос на модерацию
            cursor.execute(
                """
                INSERT INTO pending_requests (request_type, data, owner_email)
                VALUES ('update_org', %s, %s)
                """,
                (psycopg2.extras.Json(data), request.current_owner["email"]),
            )
            conn.commit()

            return jsonify({"message": "Запрос на обновление отправлен на модерацию"})
    except Exception as e:
        print(f"Ошибка обновления организации владельцем: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


# ===============================
# ИНИЦИАЛИЗАЦИЯ БАЗЫ ДАННЫХ
# ===============================
def create_tables():
    """Создание таблиц в БД"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS categories (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL UNIQUE
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS admins (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS organizations (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    main_service TEXT,
                    category_id INTEGER REFERENCES categories(id),
                    description TEXT NOT NULL,
                    address TEXT,
                    phone VARCHAR(100),
                    email VARCHAR(100),
                    website VARCHAR(255),
                    services TEXT,
                    tags TEXT,
                    contact_person_name VARCHAR(255),
                    contact_person_role VARCHAR(255),
                    contact_person_phone VARCHAR(100),
                    contact_person_email VARCHAR(255),
                    contact_person_photo_url TEXT,
                    embedding VECTOR(1024),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS organization_owners (
                    id SERIAL PRIMARY KEY,
                    full_name VARCHAR(255) NOT NULL,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    phone VARCHAR(100),
                    password_hash VARCHAR(255) NOT NULL,
                    organization_id INTEGER REFERENCES organizations(id),
                    is_verified BOOLEAN DEFAULT FALSE,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_requests (
                    id SERIAL PRIMARY KEY,
                    request_type VARCHAR(50) NOT NULL, -- new_org, claim_org, update_org, add_org
                    data JSONB NOT NULL,
                    owner_email VARCHAR(255) NOT NULL,
                    status VARCHAR(20) DEFAULT 'pending', -- pending, approved, rejected
                    reviewed_by INTEGER REFERENCES admins(id),
                    reviewed_at TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS organization_add_requests (
                    id SERIAL PRIMARY KEY,
                    organization_name VARCHAR(255) NOT NULL,
                    category_id INTEGER REFERENCES categories(id),
                    description TEXT,
                    address TEXT,
                    phone VARCHAR(100),
                    email VARCHAR(100),
                    website VARCHAR(255),
                    services TEXT,
                    tags TEXT,
                    contact_person_name VARCHAR(255),
                    contact_person_role VARCHAR(255),
                    contact_person_phone VARCHAR(100),
                    contact_person_email VARCHAR(255),
                    contact_person_photo_url TEXT,
                    requester_name VARCHAR(255) NOT NULL,
                    requester_email VARCHAR(255) NOT NULL,
                    requester_phone VARCHAR(100),
                    status VARCHAR(20) DEFAULT 'pending', -- pending, approved, rejected
                    reviewed_by INTEGER REFERENCES admins(id),
                    reviewed_at TIMESTAMP WITH TIME ZONE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """
            )

        conn.commit()
        print("✅ Все таблицы созданы или уже существуют")
    except Exception as e:
        print(f"❌ Ошибка создания таблиц: {e}")
        conn.rollback()


def create_default_admin():
    """Создание администратора по умолчанию"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            email = "admin@example.com"
            password = "admin123"
            password_hash = bcrypt.hashpw(
                password.encode("utf-8"), bcrypt.gensalt()
            ).decode("utf-8")

            cursor.execute("SELECT id FROM admins WHERE email = %s", (email,))
            if cursor.fetchone():
                print("✅ Администратор уже существует. Пропускаем создание.")
                return

            cursor.execute(
                "INSERT INTO admins (email, password_hash) VALUES (%s, %s)",
                (email, password_hash),
            )
            conn.commit()
            print("✅ Администратор создан: admin@example.com / admin123")
    except Exception as e:
        print(f"❌ Ошибка создания администратора: {e}")
        conn.rollback()


def create_test_categories():
    """Создание тестовых категорий"""
    categories = [
        "Медицина и здоровье",
        "Образование",
        "Социальная поддержка",
        "Юридические услуги",
        "Культура и досуг",
        "Спорт и фитнес",
        "Работа и карьера",
        "Финансовые услуги",
    ]

    try:
        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            for category_name in categories:
                cursor.execute(
                    "INSERT INTO categories (name) SELECT %s WHERE NOT EXISTS (SELECT 1 FROM categories WHERE name = %s)",
                    (category_name, category_name),
                )
                print(f"✅ Категория добавлена или уже существует: {category_name}")
        conn.commit()
    except Exception as e:
        print(f"❌ Ошибка создания категорий: {e}")
        conn.rollback()


def create_test_organizations():
    """Создание тестовых организаций"""
    test_organizations = [
        {
            "name": "Городская поликлиника №1",
            "main_service": "Приём терапевта",
            "category_id": 1,
            "description": "Оказание первичной медицинской помощи, профосмотры, вакцинация, лабораторные анализы.",
            "address": "г. Тюмень, ул. Ленина, 10",
            "phone": "+7 (3452) 123-45-67",
            "email": "clinic1@tyumen-med.ru",
            "website": "http://clinic1.tyumen-med.ru",
            "services": "Терапевт, педиатр, УЗИ, анализы",
            "tags": ["медицина", "поликлиника", "врачи", "бесплатно"],
            "contact_person_name": "Иванова Мария Петровна",
            "contact_person_role": "Заведующая",
            "contact_person_phone": "+7 (3452) 123-45-68",
            "contact_person_email": "ivanova@tyumen-med.ru",
            "contact_person_photo_url": "https://example.com/photos/ivanova.jpg  ",
        },
        {
            "name": "Школа программирования для детей",
            "main_service": "Курс по Python для детей",
            "category_id": 2,
            "description": "Обучение программированию детей от 8 до 16 лет, подготовка к олимпиадам, проектная работа.",
            "address": "г. Тюмень, ул. Пермякова, 50",
            "phone": "+7 (3452) 234-56-78",
            "email": "school@it-tyumen.ru",
            "website": "http://it-school-tyumen.ru",
            "services": "Python, Scratch, веб-дизайн",
            "tags": ["образование", "IT", "дети", "курсы"],
            "contact_person_name": "Сидоров Алексей Викторович",
            "contact_person_role": "Директор",
            "contact_person_phone": "+7 (3452) 234-56-79",
            "contact_person_email": "sidorov@it-tyumen.ru",
            "contact_person_photo_url": "https://example.com/photos/sidorov.jpg  ",
        },
        {
            "name": "Центр социальной помощи семьям",
            "main_service": "Выдача продуктовых наборов",
            "category_id": 3,
            "description": "Поддержка малоимущих семей, консультации, продуктовые наборы, помощь при трудоустройстве.",
            "address": "г. Тюмень, ул. Ленина, 12",
            "phone": "+7 (3452) 111-22-33",
            "email": "social-help@tyumen.gov.ru",
            "services": "Выдача продуктов, консультации, помощь инвалидам",
            "tags": ["социальная помощь", "семья", "инвалиды", "продукты"],
            "contact_person_name": "Петрова Ольга Николаевна",
            "contact_person_role": "Социальный работник",
            "contact_person_phone": "+7 (3452) 111-22-34",
            "contact_person_email": "petrova@tyumen.gov.ru",
            "contact_person_photo_url": "https://example.com/photos/petrova.jpg  ",
        },
        # --- Добавленные 17 новых организаций ---
        {
            "name": "Фитнес-клуб «Энергия»",
            "main_service": "Групповые тренировки",
            "category_id": 4,
            "description": "Фитнес, тренажёрный зал, йога, пилатес, детские секции.",
            "address": "г. Тюмень, ул. Республики, 88",
            "phone": "+7 (3452) 333-44-55",
            "email": "info@energia-fit.ru",
            "website": "http://energia-fit.ru",
            "services": "Тренажёрный зал, кардио, групповые занятия, массаж",
            "tags": ["спорт", "фитнес", "здоровье", "абонемент"],
            "contact_person_name": "Козлов Дмитрий Сергеевич",
            "contact_person_role": "Менеджер",
            "contact_person_phone": "+7 (3452) 333-44-56",
            "contact_person_email": "kozlov@energia-fit.ru",
            "contact_person_photo_url": "https://example.com/photos/kozlov.jpg"
        },
        {
            "name": "Стоматологическая клиника «Улыбка»",
            "main_service": "Лечение кариеса",
            "category_id": 1,
            "description": "Современная стоматология с использованием цифровых технологий.",
            "address": "г. Тюмень, ул. Мельникайте, 15",
            "phone": "+7 (3452) 444-55-66",
            "email": "smile@zub-tyumen.ru",
            "website": "http://ulibka-dent.ru",
            "services": "Отбеливание, имплантация, ортодонтия",
            "tags": ["стоматология", "зубы", "улыбка", "дети"],
            "contact_person_name": "Новикова Анна Валерьевна",
            "contact_person_role": "Администратор",
            "contact_person_phone": "+7 (3452) 444-55-67",
            "contact_person_email": "novikova@zub-tyumen.ru",
            "contact_person_photo_url": "https://example.com/photos/novikova.jpg"
        },
        {
            "name": "Детский сад «Солнышко»",
            "main_service": "Приём детей от 2 лет",
            "category_id": 2,
            "description": "Частный детский сад с авторской методикой развития и английским языком.",
            "address": "г. Тюмень, ул. Широтная, 102",
            "phone": "+7 (3452) 555-66-77",
            "email": "solnyshko@kids-tyumen.ru",
            "website": "http://detsad-solnyshko.ru",
            "services": "Развивающие занятия, прогулки, питание, английский",
            "tags": ["дети", "сад", "развитие", "частный"],
            "contact_person_name": "Васильева Татьяна Ивановна",
            "contact_person_role": "Заведующая",
            "contact_person_phone": "+7 (3452) 555-66-78",
            "contact_person_email": "vasilieva@kids-tyumen.ru",
            "contact_person_photo_url": "https://example.com/photos/vasilieva.jpg"
        },
        {
            "name": "Автосервис «Мотор»",
            "main_service": "Замена масла",
            "category_id": 5,
            "description": "Ремонт и обслуживание автомобилей всех марок. Диагностика, кузовной ремонт, шиномонтаж.",
            "address": "г. Тюмень, ул. Газовиков, 45",
            "phone": "+7 (3452) 666-77-88",
            "email": "service@motor-auto.ru",
            "website": "http://motor-auto.ru",
            "services": "ТО, ремонт двигателя, покраска, мойка",
            "tags": ["авто", "ремонт", "мастера", "шиномонтаж"],
            "contact_person_name": "Фролов Артём Павлович",
            "contact_person_role": "Руководитель",
            "contact_person_phone": "+7 (3452) 666-77-89",
            "contact_person_email": "frolov@motor-auto.ru",
            "contact_person_photo_url": "https://example.com/photos/frolov.jpg"
        },
        {
            "name": "Психологический центр «Гармония»",
            "main_service": "Индивидуальная консультация",
            "category_id": 6,
            "description": "Помощь в стрессах, тревожности, семейных конфликтах, кризисах.",
            "address": "г. Тюмень, ул. Харьковская, 33",
            "phone": "+7 (3452) 777-88-99",
            "email": "harmonia@psy-tyumen.ru",
            "website": "http://psy-harmonia.ru",
            "services": "Семейная терапия, коучинг, диагностика",
            "tags": ["психология", "терапия", "поддержка", "взрослые"],
            "contact_person_name": "Морозова Елена Алексеевна",
            "contact_person_role": "Психолог",
            "contact_person_phone": "+7 (3452) 777-88-00",
            "contact_person_email": "morozova@psy-tyumen.ru",
            "contact_person_photo_url": "https://example.com/photos/morozova.jpg"
        },
        {
            "name": "Библиотека им. Пушкина",
            "main_service": "Выдача книг",
            "category_id": 2,
            "description": "Городская библиотека с читальным залом, клубами и мероприятиями.",
            "address": "г. Тюмень, ул. Пушкина, 1",
            "phone": "+7 (3452) 888-99-00",
            "email": "pushkin@tyumen-lib.ru",
            "website": "http://lib-pushkin.ru",
            "services": "Читальный зал, мероприятия, электронные книги",
            "tags": ["книги", "чтение", "образование", "бесплатно"],
            "contact_person_name": "Смирнова Наталья Владимировна",
            "contact_person_role": "Заведующая",
            "contact_person_phone": "+7 (3452) 888-99-01",
            "contact_person_email": "smirnova@tyumen-lib.ru",
            "contact_person_photo_url": "https://example.com/photos/smirnova.jpg"
        },
        {
            "name": "Ресторан «Вкусно и точка»",
            "main_service": "Обеды по рабочим дням",
            "category_id": 7,
            "description": "Европейская кухня, бизнес-ланчи, банкеты, доставка.",
            "address": "г. Тюмень, ул. Ленина, 50",
            "phone": "+7 (3452) 999-00-11",
            "email": "info@vkusno-tyumen.ru",
            "website": "http://vkusno-tochka.ru",
            "services": "Банкеты, доставка, вегетарианское меню",
            "tags": ["еда", "ресторан", "банкет", "обед"],
            "contact_person_name": "Кузнецов Игорь Олегович",
            "contact_person_role": "Управляющий",
            "contact_person_phone": "+7 (3452) 999-00-12",
            "contact_person_email": "kuznetsov@vkusno-tyumen.ru",
            "contact_person_photo_url": "https://example.com/photos/kuznetsov.jpg"
        },
        {
            "name": "Юридическая консультация «Право»",
            "main_service": "Консультация по ДТП",
            "category_id": 8,
            "description": "Помощь в семейных, жилищных, уголовных и административных делах.",
            "address": "г. Тюмень, ул. Перекопская, 22",
            "phone": "+7 (3452) 101-11-22",
            "email": "pravo@law-tyumen.ru",
            "website": "http://pravo-tyumen.ru",
            "services": "Составление договоров, представительство в суде",
            "tags": ["юрист", "право", "консультация", "договоры"],
            "contact_person_name": "Лебедев Михаил Николаевич",
            "contact_person_role": "Адвокат",
            "contact_person_phone": "+7 (3452) 101-11-23",
            "contact_person_email": "lebedev@law-tyumen.ru",
            "contact_person_photo_url": "https://example.com/photos/lebedev.jpg"
        },
        {
            "name": "Студия йоги «Спокойствие»",
            "main_service": "Занятия хатха-йогой",
            "category_id": 4,
            "description": "Йога для начинающих и опытных, медитации, ретриты.",
            "address": "г. Тюмень, ул. Мельникайте, 5",
            "phone": "+7 (3452) 202-22-33",
            "email": "yoga@spokoinost.ru",
            "website": "http://yoga-spokoinost.ru",
            "services": "Медитации, растяжка, дыхательные практики",
            "tags": ["йога", "расслабление", "здоровье", "женщины"],
            "contact_person_name": "Калинина Елена Дмитриевна",
            "contact_person_role": "Инструктор",
            "contact_person_phone": "+7 (3452) 202-22-34",
            "contact_person_email": "kalinina@spokoinost.ru",
            "contact_person_photo_url": "https://example.com/photos/kalinina.jpg"
        },
        {
            "name": "Цветочный магазин «Букет»",
            "main_service": "Доставка цветов",
            "category_id": 9,
            "description": "Продажа свежих цветов, букетов, оформление свадеб и мероприятий.",
            "address": "г. Тюмень, ул. Республики, 10",
            "phone": "+7 (3452) 303-33-44",
            "email": "bouquet@flowers-tyumen.ru",
            "website": "http://bouquet-flowers.ru",
            "services": "Свадебные букеты, озеленение, торты с цветами",
            "tags": ["цветы", "подарки", "свадьба", "доставка"],
            "contact_person_name": "Григорьева Светлана Юрьевна",
            "contact_person_role": "Флорист",
            "contact_person_phone": "+7 (3452) 303-33-45",
            "contact_person_email": "grigoreva@flowers-tyumen.ru",
            "contact_person_photo_url": "https://example.com/photos/grigoreva.jpg"
        },
        {
            "name": "Танцевальная школа «Ритм»",
            "main_service": "Занятия по хип-хопу",
            "category_id": 10,
            "description": "Танцы для детей и взрослых: хип-хоп, брейк-данс, contemporary.",
            "address": "г. Тюмень, ул. Широтная, 55",
            "phone": "+7 (3452) 404-44-55",
            "email": "rhythm@dance-tyumen.ru",
            "website": "http://rhythm-dance.ru",
            "services": "Подготовка к выступлениям, танцевальные лагеря",
            "tags": ["танцы", "дети", "движение", "хип-хоп"],
            "contact_person_name": "Белов Никита Андреевич",
            "contact_person_role": "Хореограф",
            "contact_person_phone": "+7 (3452) 404-44-56",
            "contact_person_email": "belov@dance-tyumen.ru",
            "contact_person_photo_url": "https://example.com/photos/belov.jpg"
        },
        {
            "name": "Клининговая компания «Чистота»",
            "main_service": "Уборка квартир",
            "category_id": 11,
            "description": "Профессиональная уборка квартир, офисов, после ремонта.",
            "address": "г. Тюмень, ул. Газовиков, 10",
            "phone": "+7 (3452) 505-55-66",
            "email": "clean@chistota-tyumen.ru",
            "website": "http://chistota-cleaning.ru",
            "services": "Химчистка, мытьё окон, дезинфекция",
            "tags": ["уборка", "клининг", "офис", "после ремонта"],
            "contact_person_name": "Орлова Марина Викторовна",
            "contact_person_role": "Менеджер",
            "contact_person_phone": "+7 (3452) 505-55-67",
            "contact_person_email": "orlova@chistota-tyumen.ru",
            "contact_person_photo_url": "https://example.com/photos/orlova.jpg"
        },
        {
            "name": "Стоматология для детей «Малыш»",
            "main_service": "Лечение молочных зубов",
            "category_id": 1,
            "description": "Детская стоматология с дружелюбной атмосферой и анимацией.",
            "address": "г. Тюмень, ул. Пермякова, 77",
            "phone": "+7 (3452) 606-66-77",
            "email": "malysh@kids-dent.ru",
            "website": "http://malysh-dent.ru",
            "services": "Отбеливание, пломбы, профилактика",
            "tags": ["дети", "стоматология", "зубы", "игра"],
            "contact_person_name": "Соколова Анна Сергеевна",
            "contact_person_role": "Детский стоматолог",
            "contact_person_phone": "+7 (3452) 606-66-78",
            "contact_person_email": "sokolova@kids-dent.ru",
            "contact_person_photo_url": "https://example.com/photos/sokolova.jpg"
        },
        {
            "name": "Магазин электроники «ТехноМир»",
            "main_service": "Продажа смартфонов",
            "category_id": 12,
            "description": "Широкий ассортимент техники: телефоны, ноутбуки, аксессуары.",
            "address": "г. Тюмень, ул. Республики, 120",
            "phone": "+7 (3452) 707-77-88",
            "email": "tech@technomir-tyumen.ru",
            "website": "http://technomir.ru",
            "services": "Ремонт, гарантия, доставка",
            "tags": ["техника", "смартфоны", "ноутбуки", "ремонт"],
            "contact_person_name": "Волков Дмитрий Алексеевич",
            "contact_person_role": "Руководитель",
            "contact_person_phone": "+7 (3452) 707-77-89",
            "contact_person_email": "volkov@technomir-tyumen.ru",
            "contact_person_photo_url": "https://example.com/photos/volkov.jpg"
        },
        {
            "name": "Студия дизайна интерьера «Пространство»",
            "main_service": "Разработка дизайн-проекта",
            "category_id": 13,
            "description": "Дизайн квартир, домов, офисов. Авторский надзор, 3D-визуализация.",
            "address": "г. Тюмень, ул. Харьковская, 70",
            "phone": "+7 (3452) 808-88-99",
            "email": "design@prostranstvo.ru",
            "website": "http://prostranstvo-design.ru",
            "services": "Ремонт под ключ, подбор мебели, освещение",
            "tags": ["дизайн", "интерьер", "ремонт", "3D"],
            "contact_person_name": "Лебедева Ксения Игоревна",
            "contact_person_role": "Дизайнер",
            "contact_person_phone": "+7 (3452) 808-88-00",
            "contact_person_email": "lebedeva@prostranstvo.ru",
            "contact_person_photo_url": "https://example.com/photos/lebedeva.jpg"
        },
        {
            "name": "Агентство недвижимости «Ключ»",
            "main_service": "Продажа квартир",
            "category_id": 14,
            "description": "Помощь в покупке, продаже и аренде жилья и коммерческой недвижимости.",
            "address": "г. Тюмень, ул. Ленина, 75",
            "phone": "+7 (3452) 909-99-00",
            "email": "key@nedvizhimost-tyumen.ru",
            "website": "http://key-realty.ru",
            "services": "Ипотека, юридическое сопровождение, оценка",
            "tags": ["недвижимость", "квартиры", "аренда", "ипотека"],
            "contact_person_name": "Михайлов Артём Евгеньевич",
            "contact_person_role": "Риелтор",
            "contact_person_phone": "+7 (3452) 909-99-01",
            "contact_person_email": "mikhailov@nedvizhimost-tyumen.ru",
            "contact_person_photo_url": "https://example.com/photos/mikhailov.jpg"
        },
        {
            "name": "Ветеринарная клиника «ЗооЗащита»",
            "main_service": "Приём терапевта",
            "category_id": 1,
            "description": "Лечение животных, вакцинация, стерилизация, УЗИ.",
            "address": "г. Тюмень, ул. Широтная, 200",
            "phone": "+7 (3452) 100-11-22",
            "email": "veterinarian@zoozashita.ru",
            "website": "http://zoozashita-vet.ru",
            "services": "Стационар, дерматология, стоматология",
            "tags": ["ветеринар", "животные", "собаки", "кошки"],
            "contact_person_name": "Котов Павел Николаевич",
            "contact_person_role": "Ветврач",
            "contact_person_phone": "+7 (3452) 100-11-23",
            "contact_person_email": "kotov@zoozashita.ru",
            "contact_person_photo_url": "https://example.com/photos/kotov.jpg"
        }
    ]
 
    try:
        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            for org in test_organizations:
                # Генерация эмбеддинга
                text_for_embedding = generate_embedding_text(
                    name=org["name"],
                    description=org["description"],
                    services=org.get("services", ""),
                    address=org.get("address", ""),
                    tags=org.get("tags", []),
                    main_service=org.get("main_service", ""),
                )
                embedding = generate_embedding(text_for_embedding)

                cursor.execute(
                    """
                    INSERT INTO organizations (
                        name, main_service, category_id, description, address,
                        phone, email, website, services, tags,
                        contact_person_name, contact_person_role,
                        contact_person_phone, contact_person_email,
                        contact_person_photo_url, embedding
                    ) SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    WHERE NOT EXISTS (
                        SELECT 1 FROM organizations WHERE name = %s
                    );
                    """,
                    (
                        org["name"],
                        org.get("main_service"),
                        org["category_id"],
                        org["description"],
                        org["address"],
                        org.get("phone"),
                        org.get("email"),
                        org.get("website"),
                        org.get("services"),
                        org.get("tags"),
                        org.get("contact_person_name"),
                        org.get("contact_person_role"),
                        org.get("contact_person_phone"),
                        org.get("contact_person_email"),
                        org.get("contact_person_photo_url"),
                        embedding,
                        org["name"]  # для условия NOT EXISTS
                    )
                )

                print(f"✅ Организация добавлена: {org['name']}")
        conn.commit()
        print("✅ Все 30 тестовых организаций добавлены (или уже существуют).")
    except Exception as e:
        print(f"❌ Ошибка создания тестовых организаций: {e}")
        conn.rollback()


# ===============================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ===============================
if __name__ == "__main__":
    try:
        db_manager.init_pool()
        create_tables()
        create_default_admin()
        create_test_categories()
        create_test_organizations()
        app.run(host="0.0.0.0", port=5000, debug=True)
    except Exception as e:
        print(f"❌ Ошибка запуска приложения: {e}")
