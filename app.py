from sentence_transformers import SentenceTransformer
import numpy as np
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

embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

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

def create_test_organizations():
    """Создание 30 тестовых организаций"""
    test_organizations = [
        {
            "name": "Городская поликлиника №1",
            "category_id": 1,
            "description": "Оказание первичной медицинской помощи, профосмотры, консультации врачей.",
            "address": "г. Тюмень, ул. Республики, 100",
            "phone": "+7 (3452) 123-45-67",
            "email": "clinic1@tyumen-med.ru",
            "website": "http://clinic1.tyumen-med.ru",
            "services": "Терапевт, педиатр, УЗИ, анализы",
            "tags": ["медицина", "поликлиника", "врачи", "бесплатно"]
        },
        {
            "name": "Школа программирования для детей",
            "category_id": 2,
            "description": "Обучение программированию, робототехнике и цифровым технологиям для детей от 8 до 16 лет.",
            "address": "г. Тюмень, ул. Максима Горького, 55",
            "phone": "+7 (3452) 987-65-43",
            "email": "kidscode@edu-tech.ru",
            "website": "http://kidscode.tyumen-edu.ru",
            "services": "Python, Scratch, робототехника",
            "tags": ["образование", "дети", "программирование", "курсы"]
        },
        {
            "name": "Центр социальной помощи семьям",
            "category_id": 3,
            "description": "Поддержка малоимущих семей, консультации, продуктовые наборы, помощь при трудоустройстве.",
            "address": "г. Тюмень, ул. Ленина, 12",
            "phone": "+7 (3452) 111-22-33",
            "email": "social-help@tyumen.gov.ru",
            "services": "Выдача продуктов, консультации, помощь инвалидам",
            "tags": ["социальная помощь", "семья", "инвалиды", "продукты"]
        },
        {
            "name": "Юридическая консультация 'Право и порядок'",
            "category_id": 4,
            "description": "Бесплатные юридические консультации, помощь в составлении документов, представительство в суде.",
            "address": "г. Тюмень, ул. Широтная, 120",
            "phone": "+7 (3452) 444-55-66",
            "email": "law@pravo-tyumen.ru",
            "website": "http://pravo-tyumen.ru",
            "services": "Консультации, составление исков, защита прав",
            "tags": ["юрист", "консультация", "бесплатно", "суд"]
        },
        {
            "name": "Музей изобразительных искусств",
            "category_id": 5,
            "description": "Выставки живописи, скульптуры, фотоискусства. Образовательные программы и лекции.",
            "address": "г. Тюмень, ул. Челюскинцев, 35",
            "phone": "+7 (3452) 777-88-99",
            "email": "museum@tyumen-culture.ru",
            "website": "http://museum-tyumen.ru",
            "services": "Экскурсии, выставки, лекции",
            "tags": ["музей", "культура", "искусство", "экскурсии"]
        },
        {
            "name": "Фитнес-клуб 'Энергия'",
            "category_id": 6,
            "description": "Тренажёрный зал, групповые занятия, йога, персональные тренировки.",
            "address": "г. Тюмень, ул. Республики, 205",
            "phone": "+7 (3452) 555-66-77",
            "email": "fitness@energia-tyumen.ru",
            "website": "http://energia-tyumen.ru",
            "services": "Занятия с тренером, абонементы, йога",
            "tags": ["спорт", "фитнес", "тренер", "абонемент"]
        },
        {
            "name": "Центр занятости населения",
            "category_id": 7,
            "description": "Помощь в поиске работы, обучение новой профессии, консультации по трудовому праву.",
            "address": "г. Тюмень, ул. Мельникайте, 130",
            "phone": "+7 (3452) 222-33-44",
            "email": "job-center@tyumen.gov.ru",
            "website": "http://job-tyumen.ru",
            "services": "Подбор вакансий, обучение, пособия",
            "tags": ["работа", "трудоустройство", "обучение", "пособие"]
        },
        {
            "name": "Банк 'Тюмень-инвест'",
            "category_id": 8,
            "description": "Услуги для физических и юридических лиц: кредиты, вклады, переводы, открытие счёта.",
            "address": "г. Тюмень, ул. Пермякова, 88",
            "phone": "+7 (3452) 666-77-88",
            "email": "info@tyumen-invest.ru",
            "website": "http://tyumen-invest.ru",
            "services": "Кредиты, вклады, ипотека, переводы",
            "tags": ["банк", "кредит", "вклад", "финансы"]
        },
        {
            "name": "Детский сад 'Солнышко'",
            "category_id": 2,
            "description": "Дошкольное образование, развитие детей, питание, прогулки, подготовка к школе.",
            "address": "г. Тюмень, ул. Солнечная, 10",
            "phone": "+7 (3452) 111-00-22",
            "email": "detsad@solnyshko-tyumen.ru",
            "tags": ["дети", "дошкольное", "образование", "питание"]
        },
        {
            "name": "Психологический центр 'Гармония'",
            "category_id": 1,
            "description": "Консультации психолога, помощь при стрессе, тревожности, семейные консультации.",
            "address": "г. Тюмень, ул. Николая Семенова, 15",
            "phone": "+7 (3452) 333-44-55",
            "email": "psy@harmony-tyumen.ru",
            "services": "Индивидуальные и семейные консультации",
            "tags": ["психология", "поддержка", "стресс", "бесплатно"]
        },
        {
            "name": "Библиотека им. Пушкина",
            "category_id": 5,
            "description": "Читальный зал, выдачи книг, детский уголок, лекции и мероприятия.",
            "address": "г. Тюмень, ул. Ленина, 50",
            "phone": "+7 (3452) 222-11-33",
            "email": "library@tyumen-culture.ru",
            "website": "http://library-tyumen.ru",
            "services": "Выдача книг, читальный зал, мероприятия",
            "tags": ["книги", "чтение", "культура", "образование"]
        },
        {
            "name": "Автосервис 'Мотор'",
            "category_id": 7,
            "description": "Ремонт и обслуживание автомобилей, диагностика, кузовной ремонт.",
            "address": "г. Тюмень, ул. Широтная, 200",
            "phone": "+7 (3452) 444-66-77",
            "email": "auto@motor-tyumen.ru",
            "services": "Ремонт двигателя, ТО, покраска",
            "tags": ["автосервис", "ремонт", "машина", "ТО"]
        },
        {
            "name": "Аптека 'Здравствуйте'",
            "category_id": 1,
            "description": "Продажа лекарств, витаминов, средств гигиены, консультации фармацевта.",
            "address": "г. Тюмень, ул. Республики, 150",
            "phone": "+7 (3452) 555-11-22",
            "email": "apteka@zdrav-tyumen.ru",
            "services": "Лекарства, рецепты, витамины",
            "tags": ["аптека", "лекарства", "здоровье", "рецепт"]
        },
        {
            "name": "Школа иностранных языков 'Lingua'",
            "category_id": 2,
            "description": "Обучение английскому, немецкому, французскому языкам для детей и взрослых.",
            "address": "г. Тюмень, ул. Максима Горького, 80",
            "phone": "+7 (3452) 666-88-99",
            "email": "lingua@tyumen-edu.ru",
            "website": "http://lingua-tyumen.ru",
            "services": "Групповые и индивидуальные занятия",
            "tags": ["языки", "английский", "курсы", "образование"]
        },
        {
            "name": "Театр кукол 'Сказка'",
            "category_id": 5,
            "description": "Спектакли для детей и взрослых, мастер-классы, праздники.",
            "address": "г. Тюмень, ул. Челюскинцев, 50",
            "phone": "+7 (3452) 777-11-22",
            "email": "theatre@skazka-tyumen.ru",
            "website": "http://skazka-tyumen.ru",
            "services": "Спектакли, билеты, экскурсии",
            "tags": ["театр", "дети", "куклы", "искусство"]
        },
        {
            "name": "Стоматологическая клиника 'Улыбка'",
            "category_id": 1,
            "description": "Лечение зубов, протезирование, отбеливание, детская стоматология.",
            "address": "г. Тюмень, ул. Ленина, 88",
            "phone": "+7 (3452) 123-00-11",
            "email": "dentist@smile-tyumen.ru",
            "website": "http://smile-tyumen.ru",
            "services": "Лечение, удаление, импланты",
            "tags": ["стоматология", "зубы", "лечение", "дети"]
        },
        {
            "name": "Церковь Святого Николая",
            "category_id": 3,
            "description": "Богослужения, помощь нуждающимся, духовные беседы, крещение.",
            "address": "г. Тюмень, ул. Николая Семенова, 5",
            "phone": "+7 (3452) 234-56-78",
            "email": "church@nikolay-tyumen.ru",
            "services": "Молебны, помощь, беседы",
            "tags": ["церковь", "духовность", "помощь", "бесплатно"]
        },
        {
            "name": "Автомойка 'Блеск'",
            "category_id": 7,
            "description": "Мойка автомобиля, химчистка салона, полировка, обслуживание VIP-клиентов.",
            "address": "г. Тюмень, ул. Широтная, 250",
            "phone": "+7 (3452) 345-67-89",
            "email": "wash@blesk-tyumen.ru",
            "services": "Мойка, химчистка, полировка",
            "tags": ["автомойка", "машина", "чистка", "быстро"]
        },
        {
            "name": "Колледж информационных технологий",
            "category_id": 2,
            "description": "Среднее профессиональное образование по IT-специальностям, практика, трудоустройство.",
            "address": "г. Тюмень, ул. Республики, 130",
            "phone": "+7 (3452) 456-78-90",
            "email": "college@it-tyumen.ru",
            "website": "http://it-college-tyumen.ru",
            "services": "Обучение, диплом, стажировка",
            "tags": ["образование", "IT", "колледж", "работа"]
        },
        {
            "name": "Спортивный комплекс 'Олимп'",
            "category_id": 6,
            "description": "Бассейн, тренажёрный зал, танцы, боевые искусства, детские секции.",
            "address": "г. Тюмень, ул. Мельникайте, 80",
            "phone": "+7 (3452) 567-89-01",
            "email": "sport@olimp-tyumen.ru",
            "website": "http://olimp-tyumen.ru",
            "services": "Бассейн, секции, тренеры",
            "tags": ["спорт", "бассейн", "дети", "танцы"]
        },
        {
            "name": "Микрофинансовая организация 'Деньги в долг'",
            "category_id": 8,
            "description": "Выдача займов физическим лицам, рефинансирование, онлайн-заявки.",
            "address": "г. Тюмень, ул. Пермякова, 120",
            "phone": "+7 (3452) 678-90-12",
            "email": "mfo@zaim-tyumen.ru",
            "website": "http://zaim-tyumen.ru",
            "services": "Займы, рефинансирование, онлайн",
            "tags": ["займ", "деньги", "кредит", "онлайн"]
        },
        {
            "name": "Психотерапевтический кабинет 'Свет'",
            "category_id": 1,
            "description": "Помощь при депрессии, тревожности, панических атаках, семейных конфликтах.",
            "address": "г. Тюмень, ул. Николая Семенова, 25",
            "phone": "+7 (3452) 789-01-23",
            "email": "psy@svet-tyumen.ru",
            "services": "Психотерапия, кризисная помощь",
            "tags": ["психотерапия", "депрессия", "помощь", "поддержка"]
        },
        {
            "name": "Школа танцев 'Ритм'",
            "category_id": 5,
            "description": "Обучение танцам: хип-хоп, бальные, современные, детские и взрослые группы.",
            "address": "г. Тюмень, ул. Ленина, 100",
            "phone": "+7 (3452) 890-12-34",
            "email": "dance@ritm-tyumen.ru",
            "website": "http://ritm-tyumen.ru",
            "services": "Группы, индивидуальные занятия",
            "tags": ["танцы", "хип-хоп", "дети", "взрослые"]
        },
        {
            "name": "Служба доставки еды 'Быстро и вкусно'",
            "category_id": 7,
            "description": "Доставка еды из ресторанов города, круглосуточная поддержка, скидки.",
            "address": "г. Тюмень, ул. Широтная, 300",
            "phone": "+7 (3452) 901-23-45",
            "email": "delivery@fastfood-tyumen.ru",
            "website": "http://fastfood-tyumen.ru",
            "services": "Доставка, заказ онлайн, скидки",
            "tags": ["еда", "доставка", "онлайн", "быстро"]
        },
        {
            "name": "Частный детский сад 'Радуга'",
            "category_id": 2,
            "description": "Маленькие группы, индивидуальный подход, питание, развитие речи и логики.",
            "address": "г. Тюмень, ул. Солнечная, 20",
            "phone": "+7 (3452) 012-34-56",
            "email": "detsad@raduga-tyumen.ru",
            "services": "Развитие, питание, прогулки",
            "tags": ["дети", "частный", "развитие", "питание"]
        },
        {
            "name": "Клиника восстановительной медицины",
            "category_id": 1,
            "description": "Лечение после травм, реабилитация, массаж, физиотерапия.",
            "address": "г. Тюмень, ул. Республики, 180",
            "phone": "+7 (3452) 123-40-50",
            "email": "rehab@med-tyumen.ru",
            "services": "Массаж, физиотерапия, реабилитация",
            "tags": ["реабилитация", "травмы", "массаж", "лечение"]
        },
        {
            "name": "Консультационный центр для молодых родителей",
            "category_id": 3,
            "description": "Поддержка молодых мам, консультации по уходу за ребёнком, грудное вскармливание.",
            "address": "г. Тюмень, ул. Максима Горького, 100",
            "phone": "+7 (3452) 234-50-60",
            "email": "parents@support-tyumen.ru",
            "services": "Консультации, группы поддержки",
            "tags": ["молодые родители", "мамы", "поддержка", "бесплатно"]
        },
        {
            "name": "Автосалон 'Тюмень-авто'",
            "category_id": 7,
            "description": "Продажа новых и подержанных автомобилей, кредит, trade-in, сервис.",
            "address": "г. Тюмень, ул. Широтная, 350",
            "phone": "+7 (3452) 345-60-70",
            "email": "auto@tyumen-auto.ru",
            "website": "http://tyumen-auto.ru",
            "services": "Продажа авто, кредит, trade-in",
            "tags": ["авто", "продажа", "кредит", "новые"]
        },
        {
            "name": "Онлайн-школа подготовки к ЕГЭ",
            "category_id": 2,
            "description": "Подготовка к ЕГЭ по математике, русскому, физике, обществознанию.",
            "address": "г. Тюмень, онлайн",
            "phone": "+7 (3452) 456-70-80",
            "email": "ege@online-edu.ru",
            "website": "http://ege-online.ru",
            "services": "Онлайн-занятия, репетиторы, тесты",
            "tags": ["ЕГЭ", "подготовка", "онлайн", "образование"]
        },
        {
            "name": "Пункт приёма вторсырья 'Эко-плюс'",
            "category_id": 3,
            "description": "Приём пластика, бумаги, стекла, батареек. Оплата за вторсырьё.",
            "address": "г. Тюмень, ул. Ленина, 150",
            "phone": "+7 (3452) 567-80-90",
            "email": "eco@eco-plus.ru",
            "services": "Приём, оплата, экология",
            "tags": ["экология", "вторсырьё", "переработка", "деньги"]
        }
    ]

    conn = db_manager.get_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM organizations")
        count = cursor.fetchone()[0]
        if count > 0:
            logger.info(f"В БД уже есть {count} организаций. Тестовые записи не добавлены.")
            return

        for org in test_organizations:
            try:
                # Формируем текст для эмбеддинга
                text_for_embedding = generate_embedding_text(
                    name=org["name"],
                    description=org["description"],
                    services=org.get("services", ""),
                    address=org["address"],
                    tags=org.get("tags", [])
                )
                # Генерируем эмбеддинг
                embedding = generate_embedding(text_for_embedding)

                cursor.execute("""
                    INSERT INTO organizations 
                    (name, category_id, description, address, phone, email, website, services, tags, is_active, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    org["name"],
                    org["category_id"],
                    org["description"],
                    org["address"],
                    org.get("phone"),
                    org.get("email"),
                    org.get("website"),
                    org.get("services"),
                    org.get("tags", []),
                    org.get("is_active", True),
                    embedding  # <-- Векторное представление
                ))
            except Exception as e:
                logger.error(f"Ошибка при добавлении '{org['name']}': {e}")
                continue

        logger.info(f"✅ Успешно добавлено {len(test_organizations)} тестовых организаций с векторными эмбеддингами")

def generate_embedding(text: str) -> list:
    if not text or not text.strip():
        return [0.0] * 384  # размер вектора для MiniLM
    cleaned_text = ' '.join(text.strip().split())  # нормализуем пробелы
    embedding = embedding_model.encode(cleaned_text)
    return embedding.tolist()

def generate_embedding_text(name: str, description: str, services: str, address: str, tags: list) -> str:
    parts = [
        name or '',
        description or '',
        services or '',
        address or '',
        ' '.join(tags) if tags else ''
    ]
    return ' '.join(part.strip() for part in parts if part.strip())

def create_tables():
    conn = db_manager.get_connection()
    with conn.cursor() as cursor:
        # Создание таблицы категорий
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL,
                description TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        # Создание индекса для категорий
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_categories_name ON categories(name);")

        # Создание таблицы организаций
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS organizations (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                category_id INTEGER REFERENCES categories(id),
                description TEXT NOT NULL,
                address VARCHAR(500) NOT NULL,
                phone VARCHAR(50),
                email VARCHAR(100),
                website VARCHAR(255),
                services TEXT,
                tags TEXT[],
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                embedding vector(384)  -- Для векторного поиска
            );
        """)
        
        # Создание индексов для поиска организаций
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_organizations_category ON organizations(category_id);
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

def create_default_categories():
    """Создание категорий по умолчанию"""
    default_categories = [
        {'name': 'Медицинские услуги', 'description': 'Больницы, поликлиники, медицинские центры'},
        {'name': 'Образование', 'description': 'Школы, вузы, курсы, дополнительное образование'},
        {'name': 'Социальные услуги', 'description': 'Социальная защита, помощь семьям, инвалидам'},
        {'name': 'Юридические услуги', 'description': 'Юридическая помощь, консультации'},
        {'name': 'Культура и досуг', 'description': 'Театры, музеи, культурные центры'},
        {'name': 'Спорт и фитнес', 'description': 'Спортивные клубы, фитнес-центры'},
        {'name': 'Трудоустройство', 'description': 'Центры занятости, кадровые агентства'},
        {'name': 'Финансовые услуги', 'description': 'Банки, микрофинансовые организации'},
    ]
    
    conn = db_manager.get_connection()
    with conn.cursor() as cursor:
        for category in default_categories:
            cursor.execute("""
                INSERT INTO categories (name, description) 
                VALUES (%s, %s) 
                ON CONFLICT (name) DO NOTHING
            """, (category['name'], category['description']))
        
        logger.info("Категории по умолчанию созданы")

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
        'exp': datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.now(timezone.utc)
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
    """Получение списка организаций с поиском по категориям и тегам"""
    try:
        category = request.args.get('category', '').strip()
        limit = min(int(request.args.get('limit', 50)), 100)
        
        conn = db_manager.get_connection()
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            if category:
                # Поиск по названию категории или по тегам
                cursor.execute("""
                    SELECT o.id, o.name, c.name as category, o.description, o.address, 
                           o.phone, o.email, o.website, o.services
                    FROM organizations o
                    LEFT JOIN categories c ON o.category_id = c.id
                    WHERE o.is_active = TRUE 
                    AND (c.name ILIKE %s OR %s = ANY(o.tags))
                    ORDER BY o.created_at DESC
                    LIMIT %s
                """, (f'%{category}%', category, limit))
            else:
                # Все организации
                cursor.execute("""
                    SELECT o.id, o.name, c.name as category, o.description, o.address, 
                           o.phone, o.email, o.website, o.services
                    FROM organizations o
                    LEFT JOIN categories c ON o.category_id = c.id
                    WHERE o.is_active = TRUE
                    ORDER BY o.created_at DESC 
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
    """Семантический поиск организаций по смыслу (векторный поиск)"""
    try:
        query = request.args.get('q', '').strip()
        limit = min(int(request.args.get('limit', 20)), 50)
        if not query:
            return jsonify({'organizations': [], 'total': 0, 'query': query})

        # Генерируем эмбеддинг для запроса
        query_embedding = generate_embedding(query)

        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    o.id, o.name, c.name as category, o.description, o.address, 
                    o.phone, o.email, o.website, o.services, o.tags,
                    1 - (o.embedding <=> %s::vector) as relevance
                FROM organizations o
                LEFT JOIN categories c ON o.category_id = c.id
                WHERE o.is_active = TRUE 
                  AND o.embedding IS NOT NULL
                ORDER BY o.embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, limit))

            results = cursor.fetchall()
            organizations = []
            for row in results:
                org = dict(row)
                org.pop('relevance', None)  # можно оставить, если хочешь видеть релевантность
                organizations.append(org)

        return jsonify({
            'organizations': organizations,
            'total': len(organizations),
            'query': query
        })
    except Exception as e:
        logger.error(f"Ошибка семантического поиска: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/organizations/<int:org_id>', methods=['GET'])
def get_organization(org_id: int):
    """Получение детальной информации об организации"""
    try:
        conn = db_manager.get_connection()
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT o.id, o.name, c.name as category, o.description, o.address, 
                       o.phone, o.email, o.website, o.services
                FROM organizations o
                LEFT JOIN categories c ON o.category_id = c.id
                WHERE o.id = %s AND o.is_active = TRUE
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
            
            results = cursor.fetchall()
            categories = []
            
            for row in results:
                category = dict(row)
                category['slug'] = category['name'].lower().replace(' ', '-')
                categories.append(category)
            
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
            
            cursor.execute("SELECT COUNT(*) FROM categories")
            cat_count = cursor.fetchone()[0]
        
        return jsonify({
            'status': 'ok',
            'database': 'connected',
            'organizations_count': org_count,
            'categories_count': cat_count,
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

@app.route('/api/admin/categories', methods=['GET'])
@require_auth
def get_admin_categories():
    """Получение всех категорий для администратора"""
    try:
        conn = db_manager.get_connection()
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT c.id, c.name, c.description, c.created_at, c.updated_at,
                       COUNT(o.id) as organization_count
                FROM categories c
                LEFT JOIN organizations o ON c.id = o.category_id
                GROUP BY c.id, c.name, c.description, c.created_at, c.updated_at
                ORDER BY c.name
            """)
            
            results = cursor.fetchall()
            categories = []
            
            for row in results:
                category = dict(row)
                if category['created_at']:
                    category['created_at'] = category['created_at'].isoformat()
                if category['updated_at']:
                    category['updated_at'] = category['updated_at'].isoformat()
                categories.append(category)
        
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
                    RETURNING id, name, description, created_at
                """, (name, description))
                
                result = cursor.fetchone()
                category = dict(result)
                category['created_at'] = category['created_at'].isoformat()
                
                return jsonify({
                    'message': 'Категория успешно создана',
                    'category': category
                }), 201
                
            except psycopg2.IntegrityError:
                return jsonify({'error': 'Категория с таким названием уже существует'}), 400
        
    except Exception as e:
        logger.error(f"Ошибка создания категории: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

@app.route('/api/admin/organizations', methods=['GET'])
@require_auth
def get_admin_organizations():
    """Получение всех организаций для администратора"""
    try:
        page = max(int(request.args.get('page', 1)), 1)
        limit = min(int(request.args.get('limit', 20)), 100)
        offset = (page - 1) * limit

        # === Исправление: корректная обработка всех параметров ===
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

        # === Формирование запроса ===
        count_query = """
            SELECT COUNT(*) 
            FROM organizations o
            LEFT JOIN categories c ON o.category_id = c.id
            WHERE 1=1
        """
        base_query = """
            SELECT o.id, o.name, c.name as category, c.id as category_id, o.description, 
                   o.address, o.phone, o.email, o.website, o.services, o.tags, 
                   o.is_active, o.created_at, o.updated_at, o.embedding
            FROM organizations o
            LEFT JOIN categories c ON o.category_id = c.id
            WHERE 1=1
        """
        query_params = []

        if search:
            search_param = f'%{search}%'
            count_query += " AND (o.name ILIKE %s OR o.description ILIKE %s OR o.services ILIKE %s)"
            base_query += " AND (o.name ILIKE %s OR o.description ILIKE %s OR o.services ILIKE %s)"
            query_params.extend([search_param, search_param, search_param])

        if category_id is not None:
            count_query += " AND o.category_id = %s"
            base_query += " AND o.category_id = %s"
            query_params.append(category_id)

        if status == 'active':
            count_query += " AND o.is_active = TRUE"
            base_query += " AND o.is_active = TRUE"
        elif status == 'inactive':
            count_query += " AND o.is_active = FALSE"
            base_query += " AND o.is_active = FALSE"
        # 'all' не требует условия

        # === Сортировка и пагинация ===
        base_query += " ORDER BY o.created_at DESC LIMIT %s OFFSET %s"
        query_params.extend([limit, offset])

        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Общее количество
            cursor.execute(count_query, query_params[:-2])  # Без limit/offset
            total = cursor.fetchone()[0]

            # Получение данных
            cursor.execute(base_query, query_params)
            organizations = []
            for row in cursor.fetchall():
                org = dict(row)
                # Обработка дат
                if org['created_at']:
                    org['created_at'] = org['created_at'].isoformat()
                if org['updated_at']:
                    org['updated_at'] = org['updated_at'].isoformat()
                # Важно: embedding — это numpy array или list, но JSON не поддержает
                # Если нужно — преобразуем в list, но лучше убрать из админки, если не используется
                if 'embedding' in org and org['embedding'] is not None:
                    org['embedding'] = org['embedding'].tolist() if hasattr(org['embedding'], 'tolist') else list(org['embedding'])
                organizations.append(org)

        return jsonify({
            'organizations': organizations,
            'total': total,
            'page': page,
            'limit': limit,
            'pages': (total + limit - 1) // limit
        })
    except Exception as e:
        # 🔥 Теперь мы видим, что за ошибка!
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
                if not data.get(field) or not str(data.get(field)).isdigit():
                    return jsonify({'error': 'Категория обязательна для выбора'}), 400
            else:
                if not data.get(field, '').strip():
                    return jsonify({'error': f'Поле "{field}" обязательно'}), 400

        email = data.get('email', '').strip()
        if email and not validate_email(email):
            return jsonify({'error': 'Некорректный email адрес'}), 400

        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            # Проверка категории
            cursor.execute("SELECT id FROM categories WHERE id = %s", (data['category_id'],))
            if not cursor.fetchone():
                return jsonify({'error': 'Категория не существует'}), 400

            # Формируем текст и генерируем эмбеддинг
            text_for_embedding = generate_embedding_text(
                name=data['name'],
                description=data['description'],
                services=data.get('services', ''),
                address=data.get('address', ''),
                tags=data.get('tags', [])
            )
            embedding = generate_embedding(text_for_embedding)

            # Вставка с embedding сразу
            cursor.execute("""
                INSERT INTO organizations 
                (name, category_id, description, address, phone, email, website, services, tags, is_active, embedding) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, name, created_at
            """, (
                data['name'].strip(),
                data['category_id'],
                data['description'].strip(),
                data['address'].strip(),
                data.get('phone', '').strip() or None,
                email or None,
                data.get('website', '').strip() or None,
                data.get('services', '').strip() or None,
                data.get('tags', []),
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

@app.route('/api/admin/organizations/<int:org_id>', methods=['PUT'])
@require_auth
def update_organization(org_id: int):
    """Обновление организации с обновлением эмбеддинга"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Данные не предоставлены'}), 400

        # Проверка существования организации
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM organizations WHERE id = %s", (org_id,))
            org = cursor.fetchone()
            if not org:
                return jsonify({'error': 'Организация не найдена'}), 404

            # Валидация email
            email = data.get('email', '').strip()
            if email and not validate_email(email):
                return jsonify({'error': 'Некорректный email адрес'}), 400

            # Проверка категории, если указана
            if data.get('category_id'):
                cursor.execute("SELECT id FROM categories WHERE id = %s", (data['category_id'],))
                if not cursor.fetchone():
                    return jsonify({'error': 'Категория не существует'}), 400

            # Формируем текст для эмбеддинга (берём текущие значения, если не переданы)
            name = data.get('name', org['name'])
            description = data.get('description', org['description'])
            services = data.get('services', org['services'] or '')
            address = data.get('address', org['address'])
            tags = data.get('tags', org['tags'] or [])

            text_for_embedding = generate_embedding_text(name, description, services, address, tags)
            embedding = generate_embedding(text_for_embedding)

            # Обновляем всё, включая embedding
            cursor.execute("""
                UPDATE organizations 
                SET name = %s, category_id = %s, description = %s, address = %s, 
                    phone = %s, email = %s, website = %s, services = %s, 
                    tags = %s, is_active = %s, updated_at = NOW(), embedding = %s
                WHERE id = %s
                RETURNING id, name, updated_at
            """, (
                name.strip(),
                data.get('category_id', org['category_id']),
                description.strip(),
                address.strip(),
                data.get('phone', org['phone']) or None,
                email or None,
                data.get('website', org['website']) or None,
                services or None,
                tags,
                data.get('is_active', org['is_active']),
                embedding,
                org_id
            ))

            result = cursor.fetchone()
            organization = dict(result)
            organization['updated_at'] = organization['updated_at'].isoformat()

            return jsonify({
                'message': 'Организация успешно обновлена',
                'organization': organization
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
            cursor.execute("SELECT id FROM organizations WHERE id = %s", (org_id,))
            if not cursor.fetchone():
                return jsonify({'error': 'Организация не найдена'}), 404

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
            organization = dict(result)
            
            status_text = "активирована" if new_status else "деактивирована"
            
            return jsonify({
                'message': f'Организация успешно {status_text}',
                'organization': organization
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
                    COUNT(DISTINCT category_id) as total_categories
                FROM organizations
            """)
            stats = cursor.fetchone()
            
            # Статистика по категориям
            cursor.execute("""
                SELECT c.name as category, COUNT(o.id) as count
                FROM categories c
                LEFT JOIN organizations o ON c.id = o.category_id AND o.is_active = TRUE
                GROUP BY c.id, c.name
                ORDER BY count DESC
            """)
            categories = [dict(row) for row in cursor.fetchall()]
            
            # Последние организации
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

@app.route('/api/admin/organizations/<int:org_id>', methods=['GET'])
@require_auth
def get_admin_organization(org_id: int):
    """Получение детальной информации об организации для администратора"""
    try:
        conn = db_manager.get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute("""
                SELECT o.id, o.name, o.category_id, c.name as category, o.description, o.address, 
                       o.phone, o.email, o.website, o.services, o.tags, o.is_active,
                       o.created_at, o.updated_at
                FROM organizations o
                LEFT JOIN categories c ON o.category_id = c.id
                WHERE o.id = %s
            """, (org_id,))
            result = cursor.fetchone()
            if not result:
                return jsonify({'error': 'Организация не найдена'}), 404
            organization = dict(result)
            # Преобразуем даты
            if organization['created_at']:
                organization['created_at'] = organization['created_at'].isoformat()
            if organization['updated_at']:
                organization['updated_at'] = organization['updated_at'].isoformat()
            return jsonify({'organization': organization})
    except Exception as e:
        logger.error(f"Ошибка получения организации {org_id}: {e}")
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

def populate_embeddings():
    conn = db_manager.get_connection()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
        cursor.execute("""
            SELECT id, name, description, services, address, tags 
            FROM organizations 
            WHERE is_active = TRUE
        """)
        for org in cursor.fetchall():
            text = generate_embedding_text(
                name=org['name'],
                description=org['description'],
                services=org['services'] or '',
                address=org['address'],
                tags=org['tags'] or []
            )
            embedding = generate_embedding(text)
            cursor.execute("UPDATE organizations SET embedding = %s WHERE id = %s", (embedding, org['id']))
        logger.info("✅ Все эмбеддинги обновлены")

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
        create_default_categories()
        create_default_admin()

        create_test_organizations()
        
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