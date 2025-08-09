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
JWT_SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY", "your-secret-key-change-in-production")
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
                password=os.getenv("DB_PASSWORD", "postgres"),
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
        if not payload or payload.get("role") != "admin":
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
            return jsonify([dict(org) for org in organizations])
    except Exception as e:
        print(f"Ошибка получения организаций: {e}")
        return jsonify([]), 500


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
                password.encode(
                    "utf-8"), admin["password_hash"].encode("utf-8")
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


# ===============================
# АДМИН: ОРГАНИЗАЦИИ
# ===============================
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
                "SELECT id FROM categories WHERE id = %s", (
                    data["category_id"],)
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
                    "SELECT id FROM categories WHERE id = %s", (
                        data["category_id"],)
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
            cursor.execute(
                "DELETE FROM organizations WHERE id = %s", (org_id,))
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
                new_org_id = cursor.fetchone()["id"]

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
                    "SELECT id FROM organizations WHERE id = %s", (
                        data["org_id"],)
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
                cursor.execute(
                    "SELECT id FROM organizations WHERE id = %s", (org_id,))
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


# ===============================
# ВЛАДЕЛЕЦ: РЕГИСТРАЦИЯ И АВТОРИЗАЦИЯ
# ===============================
@app.route("/api/owner/register", methods=["POST"])
def register_owner():
    """Регистрация владельца (создание или привязка)"""
    try:
        data = request.get_json()

        required_fields = ["full_name", "email",
                           "password", "organization_name"]
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
                pending_data["contact_person_name"] = data.get(
                    "contact_person_name")
                pending_data["contact_person_phone"] = data.get(
                    "contact_person_phone")
                pending_data["contact_person_email"] = data.get(
                    "contact_person_email")
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

            if not bcrypt.checkpw(
                password.encode(
                    "utf-8"), owner["password_hash"].encode("utf-8")
            ):
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
                    embedding VECTOR(384),
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
                    request_type VARCHAR(50) NOT NULL, -- new_org, claim_org, update_org
                    data JSONB NOT NULL,
                    owner_email VARCHAR(255) NOT NULL,
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
                print(f"✅ Категория добавлена или уже существует: {
                      category_name}")
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
            "contact_person_photo_url": "https://example.com/photos/ivanova.jpg",
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
            "contact_person_photo_url": "https://example.com/photos/sidorov.jpg",
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
            "contact_person_photo_url": "https://example.com/photos/petrova.jpg",
        },
        # ... остальные 28 организаций (сокращено для краткости)
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
                    )
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
                        org["name"],
                    ),
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
