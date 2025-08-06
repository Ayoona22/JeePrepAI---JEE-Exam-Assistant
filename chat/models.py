from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from json import dumps, loads
from sentence_transformers import util
import os
from sqlalchemy import Float  # At top with other imports

Base = declarative_base()

class ChatSession(Base):
    __tablename__ = 'chat_sessions'
    id = Column(String, primary_key=True)  # use UUID or timestamp as string
    title = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False)
    role = Column(String(10), nullable=False) # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatSummary(Base):
    __tablename__ = 'chat_summary'

    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False)
    summary_text = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow)

class UserQuestion(Base):
    __tablename__ = 'user_questions'

    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)

# Initialize database
DB_PATH = os.path.join("data", "chat_history.db")
engine=None
Session=None

def init_database():
    """Initialize database with proper error handling"""
    global engine, Session
    
    # Ensure data directory exists with proper permissions
    os.makedirs("data", mode=0o755, exist_ok=True)
    
    # Create engine with better configuration
    engine = create_engine(
        f"sqlite:///{DB_PATH}", 
        connect_args={
            'check_same_thread': False,
            'timeout': 30
        },
        echo=False
    )
    
    try:
        # Create all tables
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        print("✅ Database tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")
        return False


def save_session(session_id, created_at):
    """Save session with error handling"""
    if not Session:
        init_database()
    
    session = Session()
    try:
        # Check if session already exists
        existing = session.query(ChatSession).filter_by(id=session_id).first()
        if not existing:
            new_session = ChatSession(id=session_id, created_at=created_at)
            session.add(new_session)
            session.commit()
            print(f"🛠️ Saving session: {session_id}")
    except Exception as e:
        print(f"❌ Error saving session: {e}")
        session.rollback()
    finally:
        session.close()

def session_exists(session_id):
    """Check if session exists with error handling"""
    if not Session:
        init_database()
    
    session = Session()
    try:
        exists = session.query(ChatSession).filter_by(id=session_id).first() is not None
        return exists
    except Exception as e:
        print(f"❌ Error checking session existence: {e}")
        return False
    finally:
        session.close()

def save_chat_message(session_id, role, content):
    """Save chat message with error handling"""
    if not Session:
        init_database()
    
    session = Session()
    try:
        new_message = ChatMessage(
            session_id=session_id, 
            role=role, 
            content=content, 
            created_at=datetime.utcnow()
        )
        session.add(new_message)
        session.commit()
    except Exception as e:
        print(f"❌ Error saving chat message: {e}")
        session.rollback()
    finally:
        session.close()

def get_last_n_messages(session_id: str, n: int = 6) -> list[tuple[str, str]]:
    """Get last N messages with error handling"""
    if not Session:
        init_database()
    
    session = Session()
    try:
        messages = (
            session.query(UserQuestion)
            .filter_by(session_id=session_id)
            .order_by(UserQuestion.id.desc())
            .limit(n)
            .all()
        )
        return list(reversed([(msg.question, msg.answer) for msg in messages]))
    except Exception as e:
        print(f"❌ Error getting messages: {e}")
        return []
    finally:
        session.close()

def save_user_question(session_id, question, answer, embedding):
    """Save user question with error handling"""
    if not Session:
        init_database()
    
    session = Session()
    try:
        entry = UserQuestion(
            session_id=session_id,
            question=question, 
            answer=answer, 
            embedding=dumps(embedding)
        )
        session.add(entry)
        session.commit()
    except Exception as e:
        print(f"❌ Error saving user question: {e}")
        session.rollback()
    finally:
        session.close()

def save_chat_summary(session_id, summary_text):
    """Save chat summary with error handling"""
    if not Session:
        init_database()
    
    session = Session()
    try:
        summary = session.query(ChatSummary).filter_by(session_id=session_id).first()
        if summary:
            print(f"🟡 Updating existing summary for session: {session_id}")
            summary.summary_text = summary_text
            summary.updated_at = datetime.utcnow()
        else:
            print(f"🟢 Creating new summary for session: {session_id}")
            summary = ChatSummary(
                session_id=session_id, 
                summary_text=summary_text, 
                updated_at=datetime.utcnow()
            )
            session.add(summary)
        session.commit()
    except Exception as e:
        print(f"❌ Error updating summary: {e}")
        session.rollback()
    finally:
        session.close()

def get_chat_summary(session_id):
    """Get chat summary with error handling"""
    if not Session:
        init_database()
    
    session = Session()
    try:
        summary = session.query(ChatSummary).filter_by(session_id=session_id).first()
        return summary.summary_text if summary else ""
    except Exception as e:
        print(f"❌ Error getting summary: {e}")
        return ""
    finally:
        session.close()

def get_total_chat_messages(session_id: str) -> int:
    """Get total chat messages with error handling"""
    if not Session:
        init_database()
    
    session = Session()
    try:
        count = session.query(ChatMessage).filter_by(session_id=session_id).count()
        return count
    except Exception as e:
        print(f"❌ Error getting message count: {e}")
        return 0
    finally:
        session.close()

def get_total_tokens(session_id: str) -> int:
    """Get total tokens with error handling"""
    if not Session:
        init_database()
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        session = Session()
        try:
            messages = session.query(UserQuestion).filter_by(session_id=session_id).all()
            
            total_tokens = 0
            for msg in messages:
                combined = f"User: {msg.question}\nBot: {msg.answer}"
                token_ids = tokenizer.encode(combined, add_special_tokens=False)
                total_tokens += len(token_ids)

            return total_tokens
        finally:
            session.close()
    except Exception as e:
        print(f"❌ Error calculating tokens: {e}")
        return 0

def clear_database():
    """Clear database with proper error handling - safer approach"""
    global engine, Session
    
    if not Session:
        init_database()
        return
    
    session = Session()
    try:
        # Instead of dropping tables, just delete all records
        print("🧹 Clearing database records...")
        session.query(ChatMessage).delete()
        session.query(UserQuestion).delete()
        session.query(ChatSummary).delete()
        session.query(ChatSession).delete()
        session.commit()
        print("✅ Database records cleared successfully")
    except Exception as e:
        print(f"❌ Error clearing database records: {e}")
        session.rollback()
        # If that fails, try the file deletion approach
        try:
            session.close()
            if engine:
                engine.dispose()
            
            if os.path.exists(DB_PATH) and os.access(DB_PATH, os.W_OK):
                os.remove(DB_PATH)
                print("🗑️ Database file removed as fallback")
                init_database()
            else:
                print("⚠️ Cannot remove database file - insufficient permissions")
        except Exception as e2:
            print(f"❌ Fallback clear also failed: {e2}")
    finally:
        session.close()

# def safe_init_database():
#     """Initialize database only if it doesn't exist or is empty"""
#     if not os.path.exists(DB_PATH):
#         init_database()
#         return True
    
#     # Check if database has tables
#     try:
#         session = Session()
#         session.query(ChatSession).first()
#         session.close()
#         print("✅ Database already initialized")
#         return True
#     except:
#         print("🔄 Database exists but needs initialization")
#         init_database()
#         return True

# # Initialize on import - safer approach
# try:
#     init_database()
# except Exception as e:
#     print(f"⚠️ Database initialization failed on import: {e}")
#     print("🔄 Will try to initialize when first used")