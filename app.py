from sentence_transformers import SentenceTransformer
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import psycopg2
import psycopg2.extras
import os
import logging
from datetime import datetime, timedelta, timezone
import bcrypt
import jwt
from functools import wraps
import re

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_model = SentenceTransformer('ai-forever/sbert_large_mt_nlu_ru')
app = Flask(__name__)
CORS(app, supports_credentials=True)

# Конфигурация
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'resource_map'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '1234'),
    'port': os.getenv('DB_PORT', '5432')
}

JWT_SECRET = os.getenv('JWT_SECRET', 'your_jwt_secret_here')
JWT_EXPIRATION_HOURS = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))

# Управление подключениями к БД
class DatabaseManager:
    def __init__(self, config):
        self.config = config

    def get_connection(self):
        return psycopg2.connect(**self.config)

db_manager = DatabaseManager(DB_CONFIG)

# Вспомогательные функции
def validate_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

def generate_embedding(text: str) -> list:
    if not text or not text.strip():
        return [0.0] * 1024
    cleaned_text = ' '.join(text.strip().split())
    embedding = embedding_model.encode(cleaned_text)
    return embedding.tolist()

def generate_embedding_text(name: str, description: str, services: str, address: str, tags: list, main_service: str = '') -> str:
    parts = [
        main_service or name or '',
        description or '',
        services or '',
        address or '',
        ' '.join(tags) if tags else ''
    ]
    return ', '.join(filter(None, parts))

# JWT и Auth
def generate_jwt_token(admin_id: int, username: str) -> str:
    payload = {
        'admin_id': admin_id,
        'username': username,
        'exp': datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.now(timezone.utc)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def verify_jwt_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return jsonify({'error': 'Токен не предоставлен'}), 401
        token = token.split(' ')[1]
        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'error': 'Неверный или истекший токен'}), 401
        request.current_admin = payload
        return f(*args, **kwargs)
    return decorated_function

# Владельцы организаций - Auth
def generate_owner_jwt_token(owner_id: int, email: str) -> str:
    payload = {
        'owner_id': owner_id,
        'email': email,
        'exp': datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.now(timezone.utc)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def verify_owner_jwt_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def require_owner_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return jsonify({'error': 'Токен не предоставлен'}), 401
        token = token.split(' ')[1]
        payload = verify_owner_jwt_token(token)
        if not payload:
            return jsonify({'error': 'Неверный или истекший токен'}), 401
        request.current_owner = payload
        return f(*args, **kwargs)
    return decorated_function

# ===============================
# API ЭНДПОИНТЫ
# ===============================

@app.route('/api/search', methods=['GET'])
def semantic_search():
    """Семантический поиск организаций"""
    try:
        query = request.args.get('q', '').strip()
        limit = min(int(request.args.get('limit', 20)), 50)
        if not query:
            return jsonify({'organizations': [], 'total': 0, 'query': query})

        query_embedding = generate_embedding(query)
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    o.id, o.name, o.main_service, c.name as category, o.description, o.address,
                    o.phone, o.email, o.website, o.services, o.tags,
                    1 - (o.embedding <=> %s::vector) as relevance
                FROM organizations o
                LEFT JOIN categories c ON o.category_id = c.id
                WHERE o.is_active = TRUE AND o.embedding IS NOT NULL
                ORDER BY o.embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, limit))
            results = cursor.fetchall()
            organizations = []
            for row in results:
                org = dict(row)
                org.pop('relevance', None)
                organizations.append(org)
            return jsonify({'organizations': organizations, 'total': len(organizations), 'query': query})
    except Exception as e:
        logger.error(f"Ошибка семантического поиска: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/organizations', methods=['GET'])
def get_organizations():
    """Получение всех активных организаций (для главной страницы)"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT o.id, o.name, o.main_service, c.name as category, o.description
                FROM organizations o
                LEFT JOIN categories c ON o.category_id = c.id
                WHERE o.is_active = TRUE
                ORDER BY o.name
            """)
            results = cursor.fetchall()
            organizations = [dict(row) for row in results]
            return jsonify({'organizations': organizations})
    except Exception as e:
        logger.error(f"Ошибка получения организаций: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/organizations/<int:org_id>', methods=['GET'])
def get_organization(org_id: int):
    """Получение детальной информации об организации"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT o.*, c.name as category_name
                FROM organizations o
                LEFT JOIN categories c ON o.category_id = c.id
                WHERE o.id = %s AND o.is_active = TRUE
            """, (org_id,))
            org = cursor.fetchone()
            if not org:
                return jsonify({'error': 'Организация не найдена'}), 404
            organization = dict(org)
            return jsonify({'organization': organization})
    except Exception as e:
        logger.error(f"Ошибка получения организации {org_id}: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Получение всех категорий"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT c.id, c.name, c.description, COUNT(o.id) as organization_count
                FROM categories c
                LEFT JOIN organizations o ON c.id = o.category_id AND o.is_active = TRUE
                GROUP BY c.id, c.name, c.description
                ORDER BY c.name
            """)
            categories = [dict(row) for row in cursor.fetchall()]
            return jsonify({'categories': categories})
    except Exception as e:
        logger.error(f"Ошибка получения категорий: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

# ===============================
# АДМИНКА
# ===============================

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    """Авторизация администратора"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Данные не предоставлены'}), 400
        username = data.get('username', '').strip()
        password = data.get('password', '')
        if not username or not password:
            return jsonify({'error': 'Имя пользователя и пароль обязательны'}), 400

        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT id, username, email, password_hash, is_active, login_attempts, locked_until
                FROM admins
                WHERE username = %s OR email = %s
            """, (username, username))
            admin = cursor.fetchone()
            if not admin:
                return jsonify({'error': 'Неверные учетные данные'}), 401
            if not admin['is_active']:
                return jsonify({'error': 'Аккаунт заблокирован'}), 401
            if admin['locked_until'] and admin['locked_until'] > datetime.now():
                return jsonify({'error': 'Аккаунт временно заблокирован. Попробуйте позже'}), 401
            if not bcrypt.checkpw(password.encode('utf-8'), admin['password_hash'].encode('utf-8')):
                new_attempts = admin['login_attempts'] + 1
                locked_until = None
                if new_attempts >= 5:
                    locked_until = datetime.now() + timedelta(minutes=30)
                cursor.execute("""
                    UPDATE admins
                    SET login_attempts = %s, locked_until = %s
                    WHERE id = %s
                """, (new_attempts, locked_until, admin['id']))
                return jsonify({'error': 'Неверные учетные данные'}), 401
            cursor.execute("""
                UPDATE admins
                SET login_attempts = 0, locked_until = NULL, last_login = NOW()
                WHERE id = %s
            """, (admin['id'],))
            token = generate_jwt_token(admin['id'], admin['username'])
            token_hash = bcrypt.hashpw(token.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            expires_at = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS)
            cursor.execute("""
                INSERT INTO admin_sessions (admin_id, token_hash, expires_at, ip_address, user_agent)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                admin['id'],
                token_hash,
                expires_at,
                request.environ.get('REMOTE_ADDR'),
                request.headers.get('User-Agent', '')[:500]
            ))
            return jsonify({
                'token': token,
                'admin': {'id': admin['id'], 'username': admin['username'], 'email': admin['email']},
                'expires_at': expires_at.isoformat()
            })
    except Exception as e:
        logger.error(f"Ошибка авторизации: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/logout', methods=['POST'])
@require_auth
def admin_logout():
    """Выход администратора"""
    token = request.headers.get('Authorization').split(' ')[1]
    token_hash = bcrypt.hashpw(token.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    conn = db_manager.get_connection()
    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM admin_sessions WHERE token_hash = %s", (token_hash,))
    return jsonify({'message': 'Выход выполнен успешно'})

@app.route('/api/admin/profile', methods=['GET'])
@require_auth
def get_admin_profile():
    """Получение профиля администратора"""
    try:
        return jsonify({'admin': request.current_admin})
    except Exception as e:
        logger.error(f"Ошибка получения профиля: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/stats', methods=['GET'])
@require_auth
def get_admin_stats():
    """Получение статистики для администратора"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT
                    COUNT(*) as total_organizations,
                    COUNT(*) FILTER (WHERE is_active = TRUE) as active_organizations,
                    COUNT(*) FILTER (WHERE is_active = FALSE) as inactive_organizations,
                    COUNT(DISTINCT category_id) as total_categories
                FROM organizations
            """)
            stats = cursor.fetchone()
            cursor.execute("""
                SELECT c.id, c.name, COUNT(o.id) as organization_count
                FROM categories c
                LEFT JOIN organizations o ON c.id = o.category_id
                GROUP BY c.id, c.name
                ORDER BY c.name
            """)
            categories = [dict(row) for row in cursor.fetchall()]
            cursor.execute("""
                SELECT o.id, o.name, c.name as category, o.created_at
                FROM organizations o
                LEFT JOIN categories c ON o.category_id = c.id
                ORDER BY o.created_at DESC
                LIMIT 5
            """)
            recent_orgs = []
            for row in cursor.fetchall():
                org = dict(row)
                org['created_at'] = org['created_at'].isoformat()
                recent_orgs.append(org)
            return jsonify({
                'stats': dict(stats),
                'categories': categories,
                'recent_organizations': recent_orgs
            })
    except Exception as e:
        logger.error(f"Ошибка получения статистики: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/organizations', methods=['GET'])
@require_auth
def get_admin_organizations():
    """Получение всех организаций для администратора"""
    try:
        page = max(int(request.args.get('page', 1)), 1)
        limit = min(int(request.args.get('limit', 20)), 100)
        offset = (page - 1) * limit

        search = request.args.get('search', '').strip()
        if not search or search.lower() == 'undefined':
            search = None
        category_id_str = request.args.get('category_id', '').strip()
        if not category_id_str or category_id_str.lower() == 'undefined':
            category_id = None
        else:
            try:
                category_id = int(category_id_str)
            except (TypeError, ValueError):
                logger.warning(f"Некорректный category_id: {category_id_str}")
                category_id = None
        status = request.args.get('status', '').strip()
        if not status or status.lower() == 'undefined':
            status = None
        elif status not in ['all', 'active', 'inactive']:
            status = None

        count_query = """SELECT COUNT(*) AS count FROM organizations o LEFT JOIN categories c ON o.category_id = c.id WHERE 1=1"""
        base_query = """SELECT o.id, o.name, o.main_service, c.name as category, c.id as category_id, o.description,
                        o.address, o.phone, o.email, o.website, o.services, o.tags, o.is_active,
                        o.created_at, o.updated_at, o.contact_person_name, o.contact_person_role,
                        o.contact_person_phone, o.contact_person_email, o.contact_person_photo_url
                        FROM organizations o LEFT JOIN categories c ON o.category_id = c.id WHERE 1=1"""

        params = []
        if search:
            count_query += " AND (o.name ILIKE %s OR o.description ILIKE %s OR o.services ILIKE %s)"
            base_query += " AND (o.name ILIKE %s OR o.description ILIKE %s OR o.services ILIKE %s)"
            like_search = f"%{search}%"
            params.extend([like_search, like_search, like_search])
        if category_id is not None:
            count_query += " AND o.category_id = %s"
            base_query += " AND o.category_id = %s"
            params.append(category_id)
        if status:
            if status == 'active':
                count_query += " AND o.is_active = TRUE"
                base_query += " AND o.is_active = TRUE"
            elif status == 'inactive':
                count_query += " AND o.is_active = FALSE"
                base_query += " AND o.is_active = FALSE"

        base_query += " ORDER BY o.name LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(count_query, params[:-2] if len(params) > 2 else params)
            total = cursor.fetchone()['count']
            cursor.execute(base_query, params)
            results = cursor.fetchall()
            organizations = []
            for org in results:
                org['created_at'] = org['created_at'].isoformat()
                if org['updated_at']:
                    org['updated_at'] = org['updated_at'].isoformat()
                organizations.append(org)
            return jsonify({
                'organizations': organizations,
                'total': total,
                'page': page,
                'limit': limit,
                'pages': (total + limit - 1) // limit
            })
    except Exception as e:
        logger.error(f"Ошибка получения организаций: {type(e).__name__}: {e}", exc_info=True)
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/organizations', methods=['POST'])
@require_auth
def create_organization():
    """Создание новой организации (с векторным эмбеддингом)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Данные не предоставлены'}), 400
        required_fields = ['name', 'category_id', 'description', 'address']
        for field in required_fields:
            if field == 'category_id':
                if not data.get('category_id'):
                    return jsonify({'error': f'Поле {field} обязательно'}), 400
            elif not data.get(field, '').strip():
                return jsonify({'error': f'Поле {field} обязательно'}), 400

        email = data.get('email', '').strip()
        if email and not validate_email(email):
            return jsonify({'error': 'Некорректный email адрес'}), 400

        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("SELECT id FROM categories WHERE id = %s", (data['category_id'],))
            if not cursor.fetchone():
                return jsonify({'error': 'Категория не существует'}), 400

            text_for_embedding = generate_embedding_text(
                name=data['name'],
                description=data['description'],
                services=data.get('services', ''),
                address=data.get('address', ''),
                tags=data.get('tags', []),
                main_service=data.get('main_service', '')
            )
            embedding = generate_embedding(text_for_embedding)

            cursor.execute("""
                INSERT INTO organizations(
                    name, main_service, category_id, description, address, phone, email, website, 
                    services, tags, contact_person_name, contact_person_role, contact_person_phone,
                    contact_person_email, contact_person_photo_url, is_active, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, name, created_at
            """, (
                data['name'].strip(),
                data.get('main_service', '').strip() or None,
                data['category_id'],
                data['description'].strip(),
                data['address'].strip(),
                data.get('phone', '').strip() or None,
                email or None,
                data.get('website', '').strip() or None,
                data.get('services', '').strip() or None,
                data.get('tags', []),
                data.get('contact_person_name', '').strip() or None,
                data.get('contact_person_role', '').strip() or None,
                data.get('contact_person_phone', '').strip() or None,
                data.get('contact_person_email', '').strip() or None,
                data.get('contact_person_photo_url', '').strip() or None,
                data.get('is_active', True),
                embedding
            ))
            result = cursor.fetchone()
            organization = dict(result)
            organization['created_at'] = organization['created_at'].isoformat()
            return jsonify({
                'message': 'Организация успешно создана',
                'organization': organization
            }), 201
    except Exception as e:
        logger.error(f"Ошибка создания организации: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/organizations/<int:org_id>', methods=['GET'])
@require_auth
def get_admin_organization(org_id: int):
    """Получение детальной информации об организации для администратора"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT o.*, c.name as category_name
                FROM organizations o
                LEFT JOIN categories c ON o.category_id = c.id
                WHERE o.id = %s
            """, (org_id,))
            org = cursor.fetchone()
            if not org:
                return jsonify({'error': 'Организация не найдена'}), 404
            organization = dict(org)
            return jsonify({'organization': organization})
    except Exception as e:
        logger.error(f"Ошибка получения организации {org_id}: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/organizations/<int:org_id>', methods=['PUT'])
@require_auth
def update_organization(org_id: int):
    """Обновление информации об организации"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Данные не предоставлены'}), 400

        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("SELECT id FROM organizations WHERE id = %s", (org_id,))
            if not cursor.fetchone():
                return jsonify({'error': 'Организация не найдена'}), 404

            email = data.get('email', '').strip()
            if email and not validate_email(email):
                return jsonify({'error': 'Некорректный email адрес'}), 400

            if data.get('category_id'):
                cursor.execute("SELECT id FROM categories WHERE id = %s", (data['category_id'],))
                if not cursor.fetchone():
                    return jsonify({'error': 'Категория не существует'}), 400

            text_for_embedding = generate_embedding_text(
                name=data.get('name', ''),
                description=data.get('description', ''),
                services=data.get('services', ''),
                address=data.get('address', ''),
                tags=data.get('tags', []),
                main_service=data.get('main_service', '')
            )
            embedding = generate_embedding(text_for_embedding)

            cursor.execute("""
                UPDATE organizations SET
                    name = %s, main_service = %s, category_id = %s, description = %s, address = %s,
                    phone = %s, email = %s, website = %s, services = %s, tags = %s,
                    contact_person_name = %s, contact_person_role = %s, contact_person_phone = %s,
                    contact_person_email = %s, contact_person_photo_url = %s, is_active = %s,
                    embedding = %s, updated_at = NOW()
                WHERE id = %s
            """, (
                data.get('name', '').strip() or None,
                data.get('main_service', '').strip() or None,
                data.get('category_id'),
                data.get('description', '').strip() or None,
                data.get('address', '').strip() or None,
                data.get('phone', '').strip() or None,
                email or None,
                data.get('website', '').strip() or None,
                data.get('services', '').strip() or None,
                data.get('tags', []),
                data.get('contact_person_name', '').strip() or None,
                data.get('contact_person_role', '').strip() or None,
                data.get('contact_person_phone', '').strip() or None,
                data.get('contact_person_email', '').strip() or None,
                data.get('contact_person_photo_url', '').strip() or None,
                data.get('is_active', True),
                embedding,
                org_id
            ))
            return jsonify({'message': 'Организация успешно обновлена'})
    except Exception as e:
        logger.error(f"Ошибка обновления организации: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/organizations/<int:org_id>', methods=['DELETE'])
@require_auth
def delete_organization(org_id: int):
    """Удаление организации"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM organizations WHERE id = %s", (org_id,))
            if cursor.rowcount == 0:
                return jsonify({'error': 'Организация не найдена'}), 404
            return jsonify({'message': 'Организация успешно удалена'})
    except Exception as e:
        logger.error(f"Ошибка удаления организации: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/categories', methods=['GET'])
@require_auth
def get_admin_categories():
    """Получение всех категорий для администратора"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT c.id, c.name, c.description, COUNT(o.id) as organization_count
                FROM categories c
                LEFT JOIN organizations o ON c.id = o.category_id
                GROUP BY c.id, c.name, c.description
                ORDER BY c.name
            """)
            categories = [dict(row) for row in cursor.fetchall()]
            return jsonify({'categories': categories})
    except Exception as e:
        logger.error(f"Ошибка получения категорий: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/categories', methods=['POST'])
@require_auth
def create_category():
    """Создание новой категории"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Данные не предоставлены'}), 400
        name = data.get('name', '').strip()
        description = data.get('description', '').strip()
        if not name:
            return jsonify({'error': 'Название категории обязательно'}), 400

        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            try:
                cursor.execute("""
                    INSERT INTO categories (name, description)
                    VALUES (%s, %s)
                    RETURNING id, name, description
                """, (name, description))
                category = dict(cursor.fetchone())
                return jsonify({
                    'message': 'Категория успешно создана',
                    'category': category
                }), 201
            except psycopg2.IntegrityError:
                return jsonify({'error': 'Категория с таким названием уже существует'}), 400
    except Exception as e:
        logger.error(f"Ошибка создания категории: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

# ===============================
# API ДЛЯ ВЛАДЕЛЬЦЕВ ОРГАНИЗАЦИЙ
# ===============================

@app.route('/api/owner/register', methods=['POST'])
def register_owner():
    """Регистрация владельца организации (заявка на модерацию)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Данные не предоставлены'}), 400

        required_fields = ['full_name', 'email', 'org_name', 'password']
        for field in required_fields:
            if not data.get(field, '').strip():
                return jsonify({'error': f'Поле {field} обязательно'}), 400

        if not validate_email(data['email']):
            return jsonify({'error': 'Некорректный email адрес'}), 400

        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM organization_owners WHERE email = %s", (data['email'],))
            if cursor.fetchone():
                return jsonify({'error': 'Пользователь с таким email уже существует'}), 400

            password_hash = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            # Проверяем, существует ли организация
            cursor.execute("SELECT id FROM organizations WHERE name ILIKE %s", (data['org_name'],))
            org_result = cursor.fetchone()

            request_type = 'claim_org' if org_result else 'new_org'
            org_id = org_result[0] if org_result else None

            pending_data = {
                'full_name': data['full_name'],
                'email': data['email'],
                'phone': data.get('phone', ''),
                'org_name': data['org_name'],
                'org_id': org_id
            }

            cursor.execute("""
                INSERT INTO pending_requests (request_type, data, owner_email)
                VALUES (%s, %s, %s)
            """, (request_type, psycopg2.extras.Json(pending_data), data['email']))

            return jsonify({'message': 'Заявка успешно отправлена на модерацию'}), 201
    except Exception as e:
        logger.error(f"Ошибка регистрации владельца: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/owner/login', methods=['POST'])
def owner_login():
    """Авторизация владельца организации"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Данные не предоставлены'}), 400
        email = data.get('email', '').strip()
        password = data.get('password', '')
        if not email or not password:
            return jsonify({'error': 'Email и пароль обязательны'}), 400

        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT id, full_name, email, password_hash, organization_id, is_verified, is_active
                FROM organization_owners
                WHERE email = %s
            """, (email,))
            owner = cursor.fetchone()
            if not owner:
                return jsonify({'error': 'Неверные учетные данные'}), 401
            if not owner['is_active']:
                return jsonify({'error': 'Аккаунт заблокирован'}), 401
            if not owner['is_verified']:
                return jsonify({'error': 'Аккаунт ожидает подтверждения администратора'}), 401
            if not bcrypt.checkpw(password.encode('utf-8'), owner['password_hash'].encode('utf-8')):
                return jsonify({'error': 'Неверные учетные данные'}), 401

            token = generate_owner_jwt_token(owner['id'], owner['email'])
            return jsonify({
                'token': token,
                'owner': {
                    'id': owner['id'],
                    'full_name': owner['full_name'],
                    'email': owner['email'],
                    'organization_id': owner['organization_id']
                }
            })
    except Exception as e:
        logger.error(f"Ошибка авторизации владельца: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/owner/organizations', methods=['GET'])
@require_owner_auth
def get_owner_organization():
    """Получение своей организации для владельца"""
    try:
        owner_id = request.current_owner['owner_id']
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT o.*
                FROM organizations o
                JOIN organization_owners oo ON o.id = oo.organization_id
                WHERE oo.id = %s
            """, (owner_id,))
            org = cursor.fetchone()
            if not org:
                return jsonify({'error': 'Организация не найдена'}), 404
            return jsonify({'organization': dict(org)})
    except Exception as e:
        logger.error(f"Ошибка получения организации владельца: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/owner/organizations', methods=['PUT'])
@require_owner_auth
def update_owner_organization():
    """Обновление своей организации владельцем (требует модерации)"""
    try:
        owner_id = request.current_owner['owner_id']
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Данные не предоставлены'}), 400

        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT o.id
                FROM organizations o
                JOIN organization_owners oo ON o.id = oo.organization_id
                WHERE oo.id = %s
            """, (owner_id,))
            org = cursor.fetchone()
            if not org:
                return jsonify({'error': 'Организация не найдена'}), 404

            org_id = org['id']
            pending_data = data.copy()
            pending_data['org_id'] = org_id

            cursor.execute("""
                INSERT INTO pending_requests (request_type, data, owner_email)
                VALUES (%s, %s, %s)
            """, ('update_org', psycopg2.extras.Json(pending_data), request.current_owner['email']))

            return jsonify({'message': 'Запрос на обновление отправлен на модерацию'})
    except Exception as e:
        logger.error(f"Ошибка обновления организации владельцем: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

# ===============================
# АДМИН: МОДЕРАЦИЯ ЗАЯВОК
# ===============================

@app.route('/api/admin/pending-requests', methods=['GET'])
@require_auth
def get_pending_requests():
    """Получение всех заявок на модерацию"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT id, request_type, data, owner_email, status, created_at
                FROM pending_requests
                ORDER BY created_at DESC
            """)
            requests = [dict(row) for row in cursor.fetchall()]
            return jsonify({'requests': requests})
    except Exception as e:
        logger.error(f"Ошибка получения заявок: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/pending-requests/<int:request_id>/approve', methods=['POST'])
@require_auth
def approve_pending_request(request_id: int):
    """Одобрение заявки на модерацию"""
    try:
        admin_id = request.current_admin['admin_id']
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT id, request_type, data, owner_email, status
                FROM pending_requests
                WHERE id = %s AND status = 'pending'
            """, (request_id,))
            req = cursor.fetchone()
            if not req:
                return jsonify({'error': 'Заявка не найдена или уже обработана'}), 404

            data = req['data']
            if req['request_type'] == 'new_org':
                # Создаем организацию
                text_for_embedding = generate_embedding_text(
                    name=data['org_name'],
                    description='',
                    services='',
                    address='',
                    tags=[],
                    main_service=''
                )
                embedding = generate_embedding(text_for_embedding)
                cursor.execute("""
                    INSERT INTO organizations (name, main_service, description, address, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    data['org_name'],
                    '', '', '', embedding
                ))
                org_id = cursor.fetchone()['id']
            elif req['request_type'] == 'claim_org':
                org_id = data['org_id']
            elif req['request_type'] == 'update_org':
                org_id = data['org_id']
                # Обновляем организацию
                update_fields = []
                update_values = []
                for key, value in data.items():
                    if key in ['name', 'main_service', 'description', 'address', 'phone', 'email', 'website', 'services', 'tags']:
                        update_fields.append(f"{key} = %s")
                        update_values.append(value)
                if update_fields:
                    text_for_embedding = generate_embedding_text(
                        name=data.get('name', ''),
                        description=data.get('description', ''),
                        services=data.get('services', ''),
                        address=data.get('address', ''),
                        tags=data.get('tags', []),
                        main_service=data.get('main_service', '')
                    )
                    embedding = generate_embedding(text_for_embedding)
                    update_fields.append("embedding = %s")
                    update_values.append(embedding)
                    update_values.append(org_id)
                    cursor.execute(f"UPDATE organizations SET {', '.join(update_fields)} WHERE id = %s", update_values)
                return jsonify({'message': 'Обновление одобрено'})

            # Создаем или обновляем владельца
            if req['request_type'] in ['new_org', 'claim_org']:
                password_hash = bcrypt.hashpw('temp_password'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                cursor.execute("""
                    INSERT INTO organization_owners (
                        full_name, email, phone, password_hash, organization_id, is_verified, is_active
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    data['full_name'],
                    data['email'],
                    data.get('phone', ''),
                    password_hash,
                    org_id,
                    True,
                    True
                ))

            # Обновляем статус заявки
            cursor.execute("""
                UPDATE pending_requests
                SET status = 'approved', reviewed_by = %s, reviewed_at = NOW()
                WHERE id = %s
            """, (admin_id, request_id))

            return jsonify({'message': 'Заявка одобрена'})
    except Exception as e:
        logger.error(f"Ошибка одобрения заявки: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/pending-requests/<int:request_id>/reject', methods=['POST'])
@require_auth
def reject_pending_request(request_id: int):
    """Отклонение заявки на модерацию"""
    try:
        admin_id = request.current_admin['admin_id']
        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE pending_requests
                SET status = 'rejected', reviewed_by = %s, reviewed_at = NOW()
                WHERE id = %s AND status = 'pending'
            """, (admin_id, request_id))
            if cursor.rowcount == 0:
                return jsonify({'error': 'Заявка не найдена или уже обработана'}), 404
            return jsonify({'message': 'Заявка отклонена'})
    except Exception as e:
        logger.error(f"Ошибка отклонения заявки: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

# ===============================
# ИНИЦИАЛИЗАЦИЯ
# ===============================

def create_tables():
    """Создание таблиц в БД при первом запуске"""
    conn = db_manager.get_connection()
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
            
            CREATE TABLE IF NOT EXISTS categories (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) UNIQUE NOT NULL,
                description TEXT
            );
            
            CREATE TABLE IF NOT EXISTS organizations (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                main_service VARCHAR(255),
                category_id INTEGER REFERENCES categories(id),
                description TEXT,
                address TEXT,
                phone VARCHAR(50),
                email VARCHAR(100),
                website VARCHAR(255),
                services TEXT,
                tags TEXT[],
                contact_person_name VARCHAR(255),
                contact_person_role VARCHAR(100),
                contact_person_phone VARCHAR(50),
                contact_person_email VARCHAR(100),
                contact_person_photo_url VARCHAR(500),
                is_active BOOLEAN DEFAULT TRUE,
                embedding VECTOR(1024),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            
            CREATE TABLE IF NOT EXISTS admins (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                login_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP WITH TIME ZONE,
                last_login TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            
            CREATE TABLE IF NOT EXISTS admin_sessions (
                id SERIAL PRIMARY KEY,
                admin_id INTEGER REFERENCES admins(id) ON DELETE CASCADE,
                token_hash VARCHAR(255) NOT NULL,
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                ip_address VARCHAR(45),
                user_agent TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            
            CREATE TABLE IF NOT EXISTS organization_owners (
                id SERIAL PRIMARY KEY,
                full_name VARCHAR(255) NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                phone VARCHAR(50),
                password_hash VARCHAR(255) NOT NULL,
                organization_id INTEGER REFERENCES organizations(id),
                is_verified BOOLEAN DEFAULT FALSE,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            
            CREATE TABLE IF NOT EXISTS pending_requests (
                id SERIAL PRIMARY KEY,
                request_type VARCHAR(20) CHECK (request_type IN ('new_org', 'claim_org', 'update_org')),
                data JSONB NOT NULL,
                owner_email VARCHAR(100),
                status VARCHAR(10) DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                reviewed_by INTEGER REFERENCES admins(id),
                reviewed_at TIMESTAMP WITH TIME ZONE
            );
        """)
        conn.commit()
        logger.info("Таблицы созданы или уже существуют")

def create_default_admin():
    """Создание администратора по умолчанию"""
    conn = db_manager.get_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM admins")
        if cursor.fetchone()[0] > 0:
            logger.info("Администраторы уже существуют")
            return
        username = "admin"
        email = "admin@tyumen-resources.ru"
        password = "admin123"
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        cursor.execute("""
            INSERT INTO admins (username, email, password_hash)
            VALUES (%s, %s, %s)
        """, (username, email, password_hash))
        logger.info("Создан администратор по умолчанию (admin/admin123)")

# ===============================
# ОБРАБОТЧИКИ ОШИБОК
# ===============================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint не найден'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

# ===============================
# ЗАПУСК ПРИЛОЖЕНИЯ
# ===============================

if __name__ == '__main__':
    try:
        create_tables()
        create_default_admin()
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Ошибка запуска приложения: {e}")
        raise
