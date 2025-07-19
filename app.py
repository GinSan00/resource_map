from flask import Flask, request, jsonify
from flask_cors import CORS
import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Конфигурация базы данных
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'dbname': os.getenv('DB_NAME', 'praktika'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '1234')
}

# Загрузка модели для векторизации
try:
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    logger.info("Модель SentenceTransformer загружена успешно")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    model = None

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
            except Exception as e:
                logger.error(f"Ошибка подключения к БД: {e}")
                raise
        return self._connection
    
    def close_connection(self):
        """Закрыть соединение с базой данных"""
        if self._connection and not self._connection.closed:
            self._connection.close()

# Инициализация менеджера БД
db_manager = DatabaseManager(DB_CONFIG)

def create_tables():
    """Создание таблиц если они не существуют"""
    conn = db_manager.get_connection()
    with conn.cursor() as cursor:
        # Создание таблицы организаций
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS organizations (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                address TEXT NOT NULL,
                contact_info JSONB,
                coordinates GEOGRAPHY(POINT, 4326),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        # Создание таблицы векторных представлений
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS organization_embeddings (
                id SERIAL PRIMARY KEY,
                organization_id INTEGER REFERENCES organizations(id) ON DELETE CASCADE,
                combined_embedding VECTOR(384),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        # Создание таблицы категорий
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                slug VARCHAR(100) UNIQUE NOT NULL,
                description TEXT,
                parent_id INTEGER REFERENCES categories(id) ON DELETE SET NULL
            );
        """)
        
        # Создание связующей таблицы организаций и категорий
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS organization_categories (
                organization_id INTEGER REFERENCES organizations(id) ON DELETE CASCADE,
                category_id INTEGER REFERENCES categories(id) ON DELETE CASCADE,
                PRIMARY KEY (organization_id, category_id)
            );
        """)
        
        # Создание функции векторного поиска
        cursor.execute("""
            CREATE OR REPLACE FUNCTION search_organizations(
                query_embedding VECTOR(384), 
                threshold FLOAT, 
                match_count INT
            )
            RETURNS TABLE(
                id INTEGER,
                name VARCHAR(255),
                description TEXT,
                address TEXT,
                contact_info JSONB,
                similarity FLOAT
            ) AS $$
            BEGIN
                RETURN QUERY
                SELECT
                    o.id,
                    o.name,
                    o.description,
                    o.address,
                    o.contact_info,
                    1 - (e.combined_embedding <=> query_embedding) AS similarity
                FROM
                    organization_embeddings e
                JOIN
                    organizations o ON e.organization_id = o.id
                WHERE
                    o.is_active = TRUE
                    AND 1 - (e.combined_embedding <=> query_embedding) > threshold
                ORDER BY
                    similarity DESC
                LIMIT
                    match_count;
            END;
            $$ LANGUAGE plpgsql;
        """)
        
        logger.info("Таблицы и функции созданы успешно")

def vectorize_text(text: str) -> Optional[List[float]]:
    """Векторизация текста с помощью SentenceTransformer"""
    if not model or not text:
        return None
    
    try:
        vector = model.encode(text)
        return vector.tolist()
    except Exception as e:
        logger.error(f"Ошибка векторизации текста: {e}")
        return None

def insert_sample_data():
    """Вставка примеров данных для демонстрации"""
    conn = db_manager.get_connection()
    
    try:
        with conn.cursor() as cursor:
            # Проверяем, есть ли уже данные
            cursor.execute("SELECT COUNT(*) FROM organizations")
            if cursor.fetchone()[0] > 0:
                logger.info("Данные уже существуют, пропускаем вставку")
                return
            
            # Вставка категорий
            categories_data = [
                ('образование', 'education', 'Образовательные учреждения'),
                ('медицина', 'healthcare', 'Медицинские учреждения'),
                ('соцзащита', 'social', 'Социальная защита'),
                ('культура', 'culture', 'Культурные учреждения'),
                ('спорт', 'sport', 'Спортивные учреждения')
            ]
            
            for slug, name, description in categories_data:
                cursor.execute(
                    "INSERT INTO categories (slug, name, description) VALUES (%s, %s, %s)",
                    (slug, name, description)
                )
            
            # Вставка организаций
            organizations_data = [
                (
                    'Детский сад №42 "Солнышко"',
                    'Муниципальное дошкольное образовательное учреждение для детей от 1.5 до 7 лет. Работаем с проблемами адаптации, буллингом среди детей, конфликтами. Имеются логопедические группы, психологическая поддержка, бассейн.',
                    'ул. Ленина, 15, Тюмень',
                    '{"phone": "+7 (3452) 123-456", "email": "ds42@tyumen.ru", "website": "http://ds42.tyumen.ru"}',
                    'образование'
                ),
                (
                    'Городская поликлиника №3',
                    'Многопрофильное лечебно-профилактическое учреждение. Помощь при стрессе, депрессии, семейных конфликтах. Взрослая и детская регистратура. Психологическая помощь.',
                    'ул. Республики, 142, Тюмень',
                    '{"phone": "+7 (3452) 234-567", "email": "pol3@tyumen-med.ru"}',
                    'медицина'
                ),
                (
                    'Центр занятости населения',
                    'Государственное учреждение, оказывающее услуги по содействию в трудоустройстве, профессиональной ориентации и переподготовке. Помощь в кризисных ситуациях, консультации по трудовым спорам.',
                    'ул. Герцена, 68, Тюмень',
                    '{"phone": "+7 (3452) 345-678", "website": "http://czn.tyumen.ru"}',
                    'соцзащита'
                ),
                (
                    'Центр психологической поддержки "Доверие"',
                    'Специализированный центр помощи людям в сложных жизненных ситуациях. Работа с буллингом, семейными конфликтами, депрессией, стрессом. Индивидуальные и групповые консультации.',
                    'ул. 50 лет Октября, 57, Тюмень',
                    '{"phone": "+7 (3452) 456-789", "email": "help@doverie-tyumen.ru", "website": "http://doverie-tyumen.ru"}',
                    'соцзащита'
                ),
                (
                    'Библиотека им. Горького',
                    'Центральная городская библиотека с обширным фондом литературы. Проводятся литературные вечера, встречи с авторами, мероприятия для борьбы со стрессом через чтение.',
                    'ул. Ленина, 2, Тюмень',
                    '{"phone": "+7 (3452) 567-890", "email": "library@tyumen.ru"}',
                    'культура'
                ),
                (
                    'Спортивный комплекс "Олимп"',
                    'Современный спортивный комплекс с бассейном, тренажерным залом, залами для групповых занятий. Помогает справиться со стрессом через физические нагрузки. Детские секции.',
                    'ул. Мельникайте, 124, Тюмень',
                    '{"phone": "+7 (3452) 678-901", "website": "http://olimp-tyumen.ru"}',
                    'спорт'
                ),
                (
                    'Школа №15 с углубленным изучением английского',
                    'Общеобразовательная школа с углубленным изучением английского языка. Работа с проблемами буллинга, психологическая поддержка учащихся. Начальная, средняя и старшая школа.',
                    'ул. Широтная, 98, Тюмень',
                    '{"phone": "+7 (3452) 789-012", "email": "school15@tyumen-edu.ru"}',
                    'образование'
                ),
                (
                    'Кризисный центр для женщин',
                    'Центр помощи женщинам, попавшим в трудную жизненную ситуацию. Помощь при домашнем насилии, психологическая поддержка, временное убежище.',
                    'ул. Комсомольская, 75, Тюмень',
                    '{"phone": "+7 (3452) 890-123", "email": "help@crisis-women.ru"}',
                    'соцзащита'
                ),
                (
                    'Молодежный центр "Перспектива"',
                    'Центр работы с молодежью. Профилактика буллинга, работа с подростками, попавшими в сложные ситуации. Досуговые и образовательные программы.',
                    'ул. Пролетарская, 35, Тюмень',
                    '{"phone": "+7 (3452) 901-234", "website": "http://mc-perspektiva.ru"}',
                    'культура'
                )
            ]
            
            # Получение ID категорий
            cursor.execute("SELECT slug, id FROM categories")
            category_ids = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Вставка организаций и их векторизация
            for name, description, address, contact_info, category_slug in organizations_data:
                # Вставка организации
                cursor.execute("""
                    INSERT INTO organizations (name, description, address, contact_info) 
                    VALUES (%s, %s, %s, %s) RETURNING id
                """, (name, description, address, contact_info))
                
                org_id = cursor.fetchone()[0]
                
                # Связывание с категорией
                if category_slug in category_ids:
                    cursor.execute("""
                        INSERT INTO organization_categories (organization_id, category_id) 
                        VALUES (%s, %s)
                    """, (org_id, category_ids[category_slug]))
                
                # Векторизация и сохранение эмбеддинга
                combined_text = f"{name} {description}"
                vector = vectorize_text(combined_text)
                
                if vector:
                    cursor.execute("""
                        INSERT INTO organization_embeddings (organization_id, combined_embedding) 
                        VALUES (%s, %s)
                    """, (org_id, vector))
            
            logger.info("Примеры данных вставлены успешно")
            
    except Exception as e:
        logger.error(f"Ошибка вставки данных: {e}")
        raise

@app.route('/api/organizations', methods=['GET'])
def get_organizations():
    """Получение списка организаций с поиском и фильтрацией"""
    try:
        query = request.args.get('query', '').strip()
        category = request.args.get('category', '').strip()
        limit = min(int(request.args.get('limit', 50)), 100)  # Максимум 100
        
        conn = db_manager.get_connection()
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            if query:
                # Векторный поиск
                query_vector = vectorize_text(query)
                if not query_vector:
                    return jsonify({'error': 'Ошибка векторизации запроса'}), 500
                
                # Использование функции векторного поиска
                cursor.execute("""
                    SELECT * FROM search_organizations(%s::vector(384), 0.1, %s)
                """, (query_vector, limit))
                
                results = cursor.fetchall()
                organizations = []
                
                for row in results:
                    org_data = dict(row)
                    org_data['similarity'] = float(org_data['similarity'])
                    org_data['distance'] = np.random.uniform(0.5, 5.0)  # Заглушка для расстояния
                    
                    # Получение категории
                    cursor.execute("""
                        SELECT c.name FROM categories c
                        JOIN organization_categories oc ON c.id = oc.category_id
                        WHERE oc.organization_id = %s LIMIT 1
                    """, (org_data['id'],))
                    
                    category_result = cursor.fetchone()
                    org_data['category'] = category_result['name'] if category_result else None
                    
                    organizations.append(org_data)
                
            else:
                # Обычный запрос с фильтрацией по категории
                base_query = """
                    SELECT DISTINCT o.id, o.name, o.description, o.address, o.contact_info
                    FROM organizations o
                """
                
                where_conditions = ["o.is_active = TRUE"]
                params = []
                
                if category:
                    base_query += """
                        JOIN organization_categories oc ON o.id = oc.organization_id
                        JOIN categories c ON oc.category_id = c.id
                    """
                    where_conditions.append("c.slug = %s")
                    params.append(category)
                
                if where_conditions:
                    base_query += " WHERE " + " AND ".join(where_conditions)
                
                base_query += " ORDER BY o.created_at DESC LIMIT %s"
                params.append(limit)
                
                cursor.execute(base_query, params)
                results = cursor.fetchall()
                
                organizations = []
                for row in results:
                    org_data = dict(row)
                    org_data['distance'] = np.random.uniform(0.5, 5.0)  # Заглушка для расстояния
                    org_data['similarity'] = 1.0  # Максимальная релевантность для обычного поиска
                    
                    # Получение категории
                    cursor.execute("""
                        SELECT c.name FROM categories c
                        JOIN organization_categories oc ON c.id = oc.category_id
                        WHERE oc.organization_id = %s LIMIT 1
                    """, (org_data['id'],))
                    
                    category_result = cursor.fetchone()
                    org_data['category'] = category_result['name'] if category_result else None
                    
                    organizations.append(org_data)
        
        return jsonify({
            'organizations': organizations,
            'total': len(organizations),
            'query': query,
            'category': category
        })
        
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
                LEFT JOIN organization_categories oc ON o.id = oc.organization_id
                LEFT JOIN categories c ON oc.category_id = c.id
                WHERE o.id = %s AND o.is_active = TRUE
            """, (org_id,))
            
            result = cursor.fetchone()
            
            if not result:
                return jsonify({'error': 'Организация не найдена'}), 404
            
            organization = dict(result)
            organization['category'] = organization.pop('category_name')
            organization['distance'] = np.random.uniform(0.5, 5.0)  # Заглушка для расстояния
            
            return jsonify({'organization': organization})
        
    except Exception as e:
        logger.error(f"Ошибка получения организации {org_id}: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Получение списка категорий"""
    try:
        conn = db_manager.get_connection()
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("SELECT id, name, slug, description FROM categories ORDER BY name")
            categories = cursor.fetchall()
            
            return jsonify({'categories': [dict(cat) for cat in categories]})
        
    except Exception as e:
        logger.error(f"Ошибка получения категорий: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/search', methods=['POST'])
def vector_search():
    """Векторный поиск организаций"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Запрос не указан'}), 400
        
        query = data['query'].strip()
        limit = min(int(data.get('limit', 10)), 50)
        threshold = float(data.get('threshold', 0.1))
        
        if not query:
            return jsonify({'error': 'Пустой запрос'}), 400
        
        # Векторизация запроса
        query_vector = vectorize_text(query)
        if not query_vector:
            return jsonify({'error': 'Ошибка векторизации запроса'}), 500
        
        conn = db_manager.get_connection()
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT * FROM search_organizations(%s::vector(384), %s, %s)
            """, (query_vector, threshold, limit))
            
            results = cursor.fetchall()
            organizations = []
            
            for row in results:
                org_data = dict(row)
                org_data['similarity'] = float(org_data['similarity'])
                org_data['distance'] = np.random.uniform(0.5, 5.0)
                
                # Получение категории
                cursor.execute("""
                    SELECT c.name FROM categories c
                    JOIN organization_categories oc ON c.id = oc.category_id
                    WHERE oc.organization_id = %s LIMIT 1
                """, (org_data['id'],))
                
                category_result = cursor.fetchone()
                org_data['category'] = category_result['name'] if category_result else None
                
                organizations.append(org_data)
        
        return jsonify({
            'organizations': organizations,
            'total': len(organizations),
            'query': query,
            'threshold': threshold
        })
        
    except Exception as e:
        logger.error(f"Ошибка векторного поиска: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка работоспособности API"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
        
        return jsonify({
            'status': 'ok',
            'database': 'connected',
            'model': 'loaded' if model else 'not loaded',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint не найден'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

if __name__ == '__main__':
    try:
        # Создание таблиц
        create_tables()
        
        # Вставка примеров данных
        insert_sample_data()
        
        logger.info("Сервер запускается...")
        
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