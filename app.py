from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import psycopg2
import psycopg2.extras
import os
import logging
from datetime import datetime, timedelta
import bcrypt
import jwt
from functools import wraps
import re

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Конфигурация
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-super-secret-key-change-in-production')
JWT_SECRET = os.getenv('JWT_SECRET', 'jwt-secret-key-change-in-production')
JWT_EXPIRATION_HOURS = 24

# Конфигурация базы данных
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'dbname': os.getenv('DB_NAME', 'resource_map'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '1234')
}

class DatabaseManager:
    def __init__(self, config: dict):
        self.config = config
        self._connection = None
    
    def get_connection(self):
        """Получить соединение с базой данных"""
        if self._connection is None or self._connection.closed:
            try:
                self._connection = psycopg2.connect(**self.config)
                self._connection.autocommit = True
                logger.info("Подключение к БД установлено")
            except Exception as e:
                logger.error(f"Ошибка подключения к БД: {e}")
                raise
        return self._connection
    
    def close_connection(self):
        """Закрыть соединение с базой данных"""
        if self._connection and not self._connection.closed:
            self._connection.close()
            logger.info("Соединение с БД закрыто")

# Инициализация менеджера БД
db_manager = DatabaseManager(DB_CONFIG)

def create_tables():
    """Создание всех необходимых таблиц"""
    conn = db_manager.get_connection()
    with conn.cursor() as cursor:
        # Создание таблицы категорий (ДОБАВЛЕНО)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL,
                description TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        # Создание индекса для категорий (ДОБАВЛЕНО)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_categories_name ON categories(name);")

        # Создание таблицы организаций
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS organizations (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                category_id INTEGER REFERENCES categories(id),  # ИЗМЕНЕНО: внешний ключ
                description TEXT NOT NULL,
                address VARCHAR(500) NOT NULL,
                phone VARCHAR(50),
                email VARCHAR(100),
                website VARCHAR(255),
                services TEXT,
                tags TEXT[],
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        # Создание индексов для поиска организаций
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_organizations_name ON organizations USING gin(to_tsvector('russian', name));
            CREATE INDEX IF NOT EXISTS idx_organizations_description ON organizations USING gin(to_tsvector('russian', description));
            CREATE INDEX IF NOT EXISTS idx_organizations_category ON organizations(category);
            CREATE INDEX IF NOT EXISTS idx_organizations_tags ON organizations USING gin(tags);
        """)
        
        # Создание таблицы администраторов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admins (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_login TIMESTAMP WITH TIME ZONE,
                login_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP WITH TIME ZONE
            );
        """)
        
        # Создание таблицы сессий администраторов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_sessions (
                id SERIAL PRIMARY KEY,
                admin_id INTEGER REFERENCES admins(id) ON DELETE CASCADE,
                token_hash VARCHAR(255) NOT NULL,
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                ip_address INET,
                user_agent TEXT
            );
        """)
        
        # Создание индексов для административных таблиц
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_admin_sessions_token ON admin_sessions(token_hash);
            CREATE INDEX IF NOT EXISTS idx_admin_sessions_expires ON admin_sessions(expires_at);
        """)
        
        logger.info("Все таблицы созданы успешно")

def create_default_admin():
    """Создание администратора по умолчанию"""
    conn = db_manager.get_connection()
    
    try:
        with conn.cursor() as cursor:
            # Проверяем, есть ли уже администраторы
            cursor.execute("SELECT COUNT(*) FROM admins")
            if cursor.fetchone()[0] > 0:
                logger.info("Администраторы уже существуют")
                return
            
            # Создаем администратора по умолчанию
            username = "admin"
            email = "admin@tyumen-resources.ru"
            password = "admin123"  # В продакшене должен быть изменен!
            
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            cursor.execute("""
                INSERT INTO admins (username, email, password_hash) 
                VALUES (%s, %s, %s)
            """, (username, email, password_hash))
            
            logger.info("Создан администратор по умолчанию (admin/admin123)")
            
    except Exception as e:
        logger.error(f"Ошибка создания администратора: {e}")
        raise

# ===============================
# JWT и AUTH функции
# ===============================

def generate_jwt_token(admin_id: int, username: str) -> str:
    """Генерация JWT токена"""
    payload = {
        'admin_id': admin_id,
        'username': username,
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def verify_jwt_token(token: str) -> dict:
    """Проверка JWT токена"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def require_auth(f):
    """Декоратор для проверки авторизации"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        
        # Проверяем заголовок Authorization
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({'error': 'Токен авторизации отсутствует'}), 401
        
        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'error': 'Недействительный или истекший токен'}), 401
        
        # Проверяем, активен ли администратор
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT id, username, email, is_active 
                FROM admins 
                WHERE id = %s AND is_active = TRUE
            """, (payload['admin_id'],))
            
            admin = cursor.fetchone()
            if not admin:
                return jsonify({'error': 'Администратор не найден или заблокирован'}), 401
        
        # Добавляем информацию об администраторе в запрос
        request.current_admin = dict(admin)
        return f(*args, **kwargs)
    
    return decorated_function

# ===============================
# Вспомогательные функции
# ===============================

def validate_email(email: str) -> bool:
    """Валидация email"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password: str) -> tuple:
    """Валидация пароля"""
    if len(password) < 6:
        return False, "Пароль должен содержать минимум 6 символов"
    
    if not re.search(r'[A-Za-z]', password):
        return False, "Пароль должен содержать буквы"
    
    if not re.search(r'\d', password):
        return False, "Пароль должен содержать цифры"
    
    return True, "Пароль валиден"

# ===============================
# ОСНОВНЫЕ МАРШРУТЫ (PUBLIC API)
# ===============================

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/admin')
def admin_index():
    return send_from_directory('static', 'admin_index.html')

@app.route('/api/organizations', methods=['GET'])
def get_organizations():
    """Получение списка организаций с поиском по тегам"""
    try:
        category = request.args.get('category', '').strip()
        limit = min(int(request.args.get('limit', 50)), 100)
        
        conn = db_manager.get_connection()
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            if category:
                # Поиск по тегам для категорий из опроса
                cursor.execute("""
                    SELECT id, name, category, description, address, phone, email, website, services
                    FROM organizations 
                    WHERE is_active = TRUE AND %s = ANY(tags)
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (category, limit))
            else:
                # Все организации
                cursor.execute("""
                    SELECT id, name, category, description, address, phone, email, website, services
                    FROM organizations 
                    WHERE is_active = TRUE
                    ORDER BY created_at DESC 
                    LIMIT %s
                """, (limit,))
            
            results = cursor.fetchall()
            organizations = [dict(row) for row in results]
        
        return jsonify({
            'organizations': organizations,
            'total': len(organizations),
            'category': category
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения организаций: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/search', methods=['GET'])
def search_organizations():
    """Поиск организаций по запросу"""
    try:
        query = request.args.get('q', '').strip()
        limit = min(int(request.args.get('limit', 20)), 50)
        
        if not query:
            return jsonify({'organizations': [], 'total': 0, 'query': query})
        
        conn = db_manager.get_connection()
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Полнотекстовый поиск по названию, описанию и услугам
            cursor.execute("""
                SELECT id, name, category, description, address, phone, email, website, services,
                       ts_rank(to_tsvector('russian', name || ' ' || description || ' ' || services), plainto_tsquery('russian', %s)) as rank
                FROM organizations 
                WHERE is_active = TRUE 
                AND (
                    to_tsvector('russian', name || ' ' || description || ' ' || services) @@ plainto_tsquery('russian', %s)
                    OR name ILIKE %s
                    OR description ILIKE %s
                    OR services ILIKE %s
                    OR category ILIKE %s
                )
                ORDER BY rank DESC, name
                LIMIT %s
            """, (query, query, f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%', limit))
            
            results = cursor.fetchall()
            organizations = []
            
            for row in results:
                org = dict(row)
                org.pop('rank', None)  # Убираем служебное поле rank
                organizations.append(org)
        
        return jsonify({
            'organizations': organizations,
            'total': len(organizations),
            'query': query
        })
        
    except Exception as e:
        logger.error(f"Ошибка поиска: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/organizations/<int:org_id>', methods=['GET'])
def get_organization(org_id: int):
    """Получение детальной информации об организации"""
    try:
        conn = db_manager.get_connection()
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT id, name, category, description, address, phone, email, website, services
                FROM organizations
                WHERE id = %s AND is_active = TRUE
            """, (org_id,))
            
            result = cursor.fetchone()
            
            if not result:
                return jsonify({'error': 'Организация не найдена'}), 404
            
            organization = dict(result)
            
            return jsonify({'organization': organization})
        
    except Exception as e:
        logger.error(f"Ошибка получения организации {org_id}: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Получение уникальных категорий"""
    try:
        conn = db_manager.get_connection()
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT DISTINCT category as name, category as slug
                FROM organizations 
                WHERE is_active = TRUE 
                ORDER BY category
            """)
            
            results = cursor.fetchall()
            categories = [dict(row) for row in results]
            
            return jsonify({'categories': categories})
        
    except Exception as e:
        logger.error(f"Ошибка получения категорий: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка работоспособности API"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM organizations WHERE is_active = TRUE")
            org_count = cursor.fetchone()[0]
        
        return jsonify({
            'status': 'ok',
            'database': 'connected',
            'organizations_count': org_count,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# ===============================
# АДМИНИСТРАТИВНЫЕ МАРШРУТЫ
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
            # Получаем администратора
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
            
            # Проверяем блокировку
            if admin['locked_until'] and admin['locked_until'] > datetime.now():
                return jsonify({'error': 'Аккаунт временно заблокирован. Попробуйте позже'}), 401
            
            # Проверяем пароль
            if not bcrypt.checkpw(password.encode('utf-8'), admin['password_hash'].encode('utf-8')):
                # Увеличиваем счетчик неудачных попыток
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
            
            # Успешная авторизация - сбрасываем счетчик попыток
            cursor.execute("""
                UPDATE admins 
                SET login_attempts = 0, locked_until = NULL, last_login = NOW()
                WHERE id = %s
            """, (admin['id'],))
            
            # Генерируем токен
            token = generate_jwt_token(admin['id'], admin['username'])
            
            # Сохраняем сессию
            token_hash = bcrypt.hashpw(token.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            expires_at = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
            
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
                'admin': {
                    'id': admin['id'],
                    'username': admin['username'],
                    'email': admin['email']
                },
                'expires_at': expires_at.isoformat()
            })
        
    except Exception as e:
        logger.error(f"Ошибка авторизации: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/logout', methods=['POST'])
@require_auth
def admin_logout():
    """Выход из системы"""
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if token:
            conn = db_manager.get_connection()
            with conn.cursor() as cursor:
                # Удаляем сессию
                cursor.execute("""
                    DELETE FROM admin_sessions 
                    WHERE admin_id = %s
                """, (request.current_admin['id'],))
        
        return jsonify({'message': 'Выход выполнен успешно'})
        
    except Exception as e:
        logger.error(f"Ошибка выхода: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/profile', methods=['GET'])
@require_auth
def get_admin_profile():
    """Получение профиля администратора"""
    try:
        return jsonify({'admin': request.current_admin})
        
    except Exception as e:
        logger.error(f"Ошибка получения профиля: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/organizations', methods=['GET'])
@require_auth
def get_admin_organizations():
    """Получение всех организаций для администратора"""
    try:
        page = int(request.args.get('page', 1))
        limit = min(int(request.args.get('limit', 20)), 100)
        search = request.args.get('search', '').strip()
        category = request.args.get('category', '').strip()
        status = request.args.get('status', 'all')  # all, active, inactive
        
        offset = (page - 1) * limit
        
        conn = db_manager.get_connection()
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Строим WHERE условие
            where_conditions = []
            params = []
            
            if search:
                where_conditions.append("""
                    (name ILIKE %s OR description ILIKE %s OR services ILIKE %s)
                """)
                search_param = f'%{search}%'
                params.extend([search_param, search_param, search_param])
            
            if category:
                where_conditions.append("category = %s")
                params.append(category)
            
            if status == 'active':
                where_conditions.append("is_active = TRUE")
            elif status == 'inactive':
                where_conditions.append("is_active = FALSE")
            
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            # Получаем общее количество
            cursor.execute(f"""
                SELECT COUNT(*) FROM organizations {where_clause}
            """, params)
            total = cursor.fetchone()[0]
            
            # Получаем организации
            cursor.execute(f"""
                SELECT id, name, category, description, address, phone, email, website, 
                       services, tags, is_active, created_at, updated_at
                FROM organizations 
                {where_clause}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """, params + [limit, offset])
            
            organizations = [dict(row) for row in cursor.fetchall()]
            
            # Преобразуем даты в строки
            for org in organizations:
                if org['created_at']:
                    org['created_at'] = org['created_at'].isoformat()
                if org['updated_at']:
                    org['updated_at'] = org['updated_at'].isoformat()
        
        return jsonify({
            'organizations': organizations,
            'total': total,
            'page': page,
            'limit': limit,
            'pages': (total + limit - 1) // limit
        })
        
    except Exception as e:
        logger.error(f"Ошибка получения организаций: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/organizations', methods=['POST'])
@require_auth
def create_organization():
    """Создание новой организации"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Данные не предоставлены'}), 400
        
        # Валидация обязательных полей
        required_fields = ['name', 'category', 'description', 'address']
        for field in required_fields:
            if not data.get(field, '').strip():
                return jsonify({'error': f'Поле "{field}" обязательно для заполнения'}), 400
        
        # Валидация email
        email = data.get('email', '').strip()
        if email and not validate_email(email):
            return jsonify({'error': 'Некорректный email адрес'}), 400
        
        conn = db_manager.get_connection()
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                INSERT INTO organizations 
                (name, category, description, address, phone, email, website, services, tags, is_active) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, name, category, created_at
            """, (
                data['name'].strip(),
                data['category'].strip(),
                data['description'].strip(),
                data['address'].strip(),
                data.get('phone', '').strip() or None,
                email or None,
                data.get('website', '').strip() or None,
                data.get('services', '').strip() or None,
                data.get('tags', []),
                data.get('is_active', True)
            ))
            
            result = cursor.fetchone()
            
            return jsonify({
                'message': 'Организация успешно создана',
                'organization': dict(result)
            }), 201
        
    except Exception as e:
        logger.error(f"Ошибка создания организации: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/organizations/<int:org_id>', methods=['PUT'])
@require_auth
def update_organization(org_id: int):
    """Обновление организации"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Данные не предоставлены'}), 400
        
        # Валидация email
        email = data.get('email', '').strip()
        if email and not validate_email(email):
            return jsonify({'error': 'Некорректный email адрес'}), 400
        
        conn = db_manager.get_connection()
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Проверяем существование организации
            cursor.execute("SELECT id FROM organizations WHERE id = %s", (org_id,))
            if not cursor.fetchone():
                return jsonify({'error': 'Организация не найдена'}), 404
            
            # Обновляем организацию
            cursor.execute("""
                UPDATE organizations 
                SET name = %s, category = %s, description = %s, address = %s, 
                    phone = %s, email = %s, website = %s, services = %s, 
                    tags = %s, is_active = %s, updated_at = NOW()
                WHERE id = %s
                RETURNING id, name, category, updated_at
            """, (
                data.get('name', '').strip(),
                data.get('category', '').strip(),
                data.get('description', '').strip(),
                data.get('address', '').strip(),
                data.get('phone', '').strip() or None,
                email or None,
                data.get('website', '').strip() or None,
                data.get('services', '').strip() or None,
                data.get('tags', []),
                data.get('is_active', True),
                org_id
            ))
            
            result = cursor.fetchone()
            
            return jsonify({
                'message': 'Организация успешно обновлена',
                'organization': dict(result)
            })
        
    except Exception as e:
        logger.error(f"Ошибка обновления организации {org_id}: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/organizations/<int:org_id>', methods=['DELETE'])
@require_auth
def delete_organization(org_id: int):
    """Удаление организации"""
    try:
        conn = db_manager.get_connection()
        
        with conn.cursor() as cursor:
            # Проверяем существование организации
            cursor.execute("SELECT id FROM organizations WHERE id = %s", (org_id,))
            if not cursor.fetchone():
                return jsonify({'error': 'Организация не найдена'}), 404
            
            # Удаляем организацию
            cursor.execute("DELETE FROM organizations WHERE id = %s", (org_id,))
            
            return jsonify({'message': 'Организация успешно удалена'})
        
    except Exception as e:
        logger.error(f"Ошибка удаления организации {org_id}: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/organizations/<int:org_id>/toggle', methods=['PATCH'])
@require_auth
def toggle_organization_status(org_id: int):
    """Переключение статуса активности организации"""
    try:
        conn = db_manager.get_connection()
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Получаем текущий статус
            cursor.execute("SELECT id, is_active FROM organizations WHERE id = %s", (org_id,))
            org = cursor.fetchone()
            
            if not org:
                return jsonify({'error': 'Организация не найдена'}), 404
            
            # Переключаем статус
            new_status = not org['is_active']
            cursor.execute("""
                UPDATE organizations 
                SET is_active = %s, updated_at = NOW()
                WHERE id = %s
                RETURNING id, name, is_active
            """, (new_status, org_id))
            
            result = cursor.fetchone()
            
            status_text = "активирована" if new_status else "деактивирована"
            
            return jsonify({
                'message': f'Организация успешно {status_text}',
                'organization': dict(result)
            })
        
    except Exception as e:
        logger.error(f"Ошибка переключения статуса организации {org_id}: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/stats', methods=['GET'])
@require_auth
def get_admin_stats():
    """Получение статистики для администратора"""
    try:
        conn = db_manager.get_connection()
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Общая статистика
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_organizations,
                    COUNT(*) FILTER (WHERE is_active = TRUE) as active_organizations,
                    COUNT(*) FILTER (WHERE is_active = FALSE) as inactive_organizations,
                    COUNT(DISTINCT category) as total_categories
                FROM organizations
            """)
            stats = cursor.fetchone()
            
            # Статистика по категориям
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM organizations
                WHERE is_active = TRUE
                GROUP BY category
                ORDER BY count DESC
            """)
            categories = [dict(row) for row in cursor.fetchall()]
            
            # Последние организации
            cursor.execute("""
                SELECT id, name, category, created_at
                FROM organizations
                ORDER BY created_at DESC
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
        # Создание всех таблиц
        create_tables()
        
        # Создание администратора по умолчанию
        create_default_admin()
        
        logger.info("Объединенный сервер запускается...")
        logger.info("Доступные endpoints:")
        logger.info("  - Public API: http://localhost:5000/api/")
        logger.info("  - Admin API: http://localhost:5000/api/admin/")
        logger.info("  - Frontend: http://localhost:5000/")
        logger.info("  - Admin Panel: http://localhost:5000/admin")
        logger.info("  - Admin credentials: admin/admin123")
        
        # Запуск сервера
        app.run(
            host='0.0.0.0',
            port=int(os.getenv('PORT', 5000)),
            debug=os.getenv('DEBUG', 'False').lower() == 'true'
        )
        
    except Exception as e:
        logger.error(f"Ошибка запуска сервера: {e}")
    finally:
        db_manager.close_connection()