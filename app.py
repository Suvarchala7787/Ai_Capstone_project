import streamlit as st
import sqlite3
import hashlib
import secrets
import time
from pathlib import Path
from datetime import datetime
from google import genai
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from google.genai import types

# =========================================================
# 1. APP CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Concept Detective AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_PATH = Path(__file__).with_name("concept_detective.db")

# Imagery for interface
HERO_IMAGE = (
    "https://images.unsplash.com/photo-1521737711867-e3b97375f902"
    "?auto=format&fit=crop&w=1800&q=85"
)

STUDY_IMAGE = (
    "https://images.unsplash.com/photo-1516321318423-f06f85e504b3"
    "?auto=format&fit=crop&w=1200&q=85"
)


# =========================================================
# 2. SUBJECTS AND TOPICS
# =========================================================

SUBJECT_TOPICS = {
    "Java": [
        "Introduction to Java",
        "Variables and Data Types",
        "Operators",
        "If-Else Statements",
        "Loops",
        "Arrays",
        "Strings",
        "Methods",
        "Object-Oriented Programming"
    ],
    "Python": [
        "Introduction to Python",
        "Variables and Data Types",
        "Operators",
        "Conditional Statements",
        "Loops",
        "Lists and Tuples",
        "Strings",
        "Functions",
        "Classes and Objects"
    ],
    "SQL and DBMS": [
        "Introduction to Databases",
        "Tables and Data Types",
        "SELECT Queries",
        "WHERE and Filtering",
        "Aggregate Functions",
        "GROUP BY and HAVING",
        "SQL Joins",
        "Keys and Constraints",
        "Normalization"
    ],
    "Data Structures and Algorithms": [
        "Introduction to DSA",
        "Arrays",
        "Strings",
        "Searching",
        "Sorting",
        "Linked Lists",
        "Stacks and Queues",
        "Hashing",
        "Trees"
    ],
    "Artificial Intelligence": [
        "Introduction to AI",
        "Types of AI",
        "Introduction to Machine Learning",
        "Data and Features",
        "Supervised Learning",
        "Unsupervised Learning",
        "Model Evaluation",
        "Neural Networks"
    ],
    "Mathematics": [
        "Number Systems",
        "Fractions and Decimals",
        "Algebra Basics",
        "Linear Equations",
        "Functions",
        "Geometry",
        "Probability",
        "Statistics"
    ],
    "Operating Systems": [
        "Introduction to Operating Systems",
        "Processes and Threads",
        "CPU Scheduling",
        "Process Synchronization",
        "Deadlocks",
        "Memory Management",
        "Virtual Memory",
        "File Systems"
    ],
    "Computer Networks": [
        "Introduction to Networks",
        "OSI Model",
        "TCP/IP Model",
        "IP Addressing",
        "Network Devices",
        "TCP and UDP",
        "DNS and HTTP",
        "Network Security Basics"
    ]
}


# =========================================================
# 3. LESSON CONTENT
# =========================================================

LESSONS = {
    "Java": {
        "Introduction to Java": {
            "definition": (
                "Java is a programming language used to build "
                "applications, backend systems and many other "
                "software solutions."
            ),
            "why": (
                "Programming languages let us give instructions "
                "to computers. Java provides a structured way "
                "to write and run those instructions."
            ),
            "example": (
                'class Main {\n'
                '    public static void main(String[] args) {\n'
                '        System.out.println("Hello World");\n'
                '    }\n'
                '}'
            ),
            "dry_run": (
                "1. The program starts in main().\n"
                "2. System.out.println executes.\n"
                '3. The text Hello World appears on the screen.'
            ),
            "plan": [
                "What programming means",
                "What Java is used for",
                "How a Java program is structured",
                "Understanding the first program",
                "A short practice quiz"
            ]
        },
        "Variables and Data Types": {
            "definition": (
                "A variable is a named place in a program where "
                "a value can be stored. A data type describes "
                "what kind of value the variable can hold."
            ),
            "why": (
                "Imagine a labelled box named age. Instead of "
                "remembering a number without context, we store "
                "it under the name age and use it later."
            ),
            "example": (
                "int age = 20;\n"
                "double price = 45.5;\n"
                'String name = "Ravi";\n'
                "boolean passed = true;"
            ),
            "dry_run": (
                "1. int means the variable stores an integer.\n"
                "2. age is the variable name.\n"
                "3. 20 is the value assigned to age.\n"
                "4. Later, the program can use age in calculations."
            ),
            "plan": [
                "What a variable is",
                "Declaration and initialization",
                "Integer, decimal and boolean values",
                "String variables",
                "Reading simple Java statements",
                "A short practice quiz"
            ]
        },
        "Loops": {
            "definition": (
                "A loop repeats a block of instructions while "
                "a condition or repetition rule allows it."
            ),
            "why": (
                "If you want to print numbers from 1 to 5, "
                "a loop avoids writing the same print statement "
                "five separate times."
            ),
            "example": (
                "for (int i = 1; i <= 5; i++) {\n"
                "    System.out.println(i);\n"
                "}"
            ),
            "dry_run": (
                "i = 1: print 1\n"
                "i = 2: print 2\n"
                "i = 3: print 3\n"
                "i = 4: print 4\n"
                "i = 5: print 5\n"
                "i becomes 6, so the condition is false and the loop ends."
            ),
            "plan": [
                "Why loops are useful",
                "Understanding the for loop",
                "Initialization, condition and update",
                "Tracing a loop one iteration at a time",
                "Common loop mistakes",
                "A short practice quiz"
            ]
        },
        "Arrays": {
            "definition": (
                "An array stores multiple values of the same "
                "declared element type under one variable name."
            ),
            "why": (
                "Instead of creating separate variables for "
                "five marks, an array lets us store them together "
                "and access each value by its index."
            ),
            "example": (
                "int[] marks = {80, 75, 90, 65};\n"
                "System.out.println(marks[0]);"
            ),
            "dry_run": (
                "marks[0] is 80.\n"
                "marks[1] is 75.\n"
                "marks[2] is 90.\n"
                "marks[3] is 65.\n"
                "Java array indexing starts at zero."
            ),
            "plan": [
                "Why arrays are needed",
                "Creating and initializing an array",
                "Understanding indexes",
                "Accessing array elements",
                "Traversing an array with a loop",
                "A short practice quiz"
            ]
        },
        "If-Else Statements": {
            "definition": (
                "An if-else statement lets a program choose "
                "between actions based on a condition."
            ),
            "why": (
                "For example, a program can check whether a "
                "student's mark is at least 40 before deciding "
                "whether the student has passed."
            ),
            "example": (
                "int marks = 65;\n"
                "if (marks >= 40) {\n"
                '    System.out.println("Pass");\n'
                "} else {\n"
                '    System.out.println("Fail");\n'
                "}"
            ),
            "dry_run": (
                "1. marks contains 65.\n"
                "2. Java checks whether 65 >= 40.\n"
                "3. The condition is true.\n"
                "4. Pass is printed."
            ),
            "plan": [
                "What a condition means",
                "Understanding comparison operators",
                "The if block",
                "The else block",
                "Tracing true and false conditions",
                "A short practice quiz"
            ]
        }
    },
    "Python": {
        "Introduction to Python": {
            "definition": (
                "Python is a general-purpose programming language "
                "known for readable syntax."
            ),
            "why": (
                "Python is used in automation, web development, "
                "data analysis and machine learning."
            ),
            "example": 'print("Hello World")',
            "dry_run": (
                "Python executes the print statement and displays "
                "Hello World in the output."
            ),
            "plan": [
                "What programming means",
                "What Python is used for",
                "Understanding Python syntax",
                "Running a first program",
                "A short practice quiz"
            ]
        },
        "Variables and Data Types": {
            "definition": (
                "A variable is a name used to refer to a value. "
                "Python determines the type of a value at runtime."
            ),
            "why": (
                "Variables let programs remember information "
                "and use it later."
            ),
            "example": (
                "age = 20\n"
                'name = "Ravi"\n'
                "price = 45.5\n"
                "passed = True"
            ),
            "dry_run": (
                "age refers to the integer 20.\n"
                "name refers to the text Ravi.\n"
                "price refers to the decimal value 45.5.\n"
                "passed refers to the Boolean value True."
            ),
            "plan": [
                "What variables are",
                "Assigning values",
                "Integers and decimal numbers",
                "Strings and Boolean values",
                "Reading variable assignments",
                "A short practice quiz"
            ]
        },
        "Loops": {
            "definition": (
                "Loops repeat instructions over a sequence or "
                "while a condition remains true."
            ),
            "why": "Loops reduce repeated code.",
            "example": (
                "for i in range(1, 6):\n"
                "    print(i)"
            ),
            "dry_run": (
                "range(1, 6) produces 1, 2, 3, 4 and 5.\n"
                "The loop prints each value in order."
            ),
            "plan": [
                "Why loops are useful",
                "The for loop",
                "The range function",
                "Tracing loop iterations",
                "Common mistakes",
                "A short practice quiz"
            ]
        },
        "Lists and Tuples": {
            "definition": (
                "A list stores an ordered collection of values "
                "and can be changed. A tuple is also ordered "
                "but is immutable."
            ),
            "why": "Collections help organise related data.",
            "example": (
                "marks = [80, 75, 90]\n"
                "print(marks[0])"
            ),
            "dry_run": (
                "The first element is at index 0.\n"
                "marks[0] returns 80."
            ),
            "plan": [
                "What collections are",
                "Creating a list",
                "Accessing elements by index",
                "Changing list values",
                "How tuples differ from lists",
                "A short practice quiz"
            ]
        }
    },
    "SQL and DBMS": {
        "SELECT Queries": {
            "definition": (
                "SELECT is an SQL command used to retrieve data "
                "from one or more database tables."
            ),
            "why": (
                "A database may contain thousands of rows. "
                "SELECT helps retrieve the information we need."
            ),
            "example": (
                "SELECT name, marks\n"
                "FROM Students;"
            ),
            "dry_run": (
                "1. SQL looks at the Students table.\n"
                "2. It selects the name and marks columns.\n"
                "3. The matching rows are returned."
            ),
            "plan": [
                "What a database table contains",
                "The SELECT command",
                "Choosing specific columns",
                "Using SELECT *",
                "Reading query results",
                "A short practice quiz"
            ]
        },
        "SQL Joins": {
            "definition": (
                "A JOIN combines related rows from tables using "
                "a specified relationship or join condition."
            ),
            "why": (
                "Student details and department details may be "
                "stored in separate tables. A join can combine "
                "their related information."
            ),
            "example": (
                "SELECT Students.name, Departments.department_name\n"
                "FROM Students\n"
                "INNER JOIN Departments\n"
                "ON Students.department_id = Departments.id;"
            ),
            "dry_run": (
                "1. SQL reads the Students table.\n"
                "2. It matches department IDs with Departments.id.\n"
                "3. INNER JOIN returns rows that satisfy the join condition."
            ),
            "plan": [
                "Why tables are separated",
                "What a join does",
                "Understanding INNER JOIN",
                "Understanding the join condition",
                "Reading joined results",
                "A short practice quiz"
            ]
        },
        "Aggregate Functions": {
            "definition": (
                "Aggregate functions calculate a single summary "
                "value from a set of rows."
            ),
            "why": (
                "They help calculate totals, averages, counts, "
                "minimums and maximums."
            ),
            "example": (
                "SELECT AVG(marks), MAX(marks), MIN(marks)\n"
                "FROM Students;"
            ),
            "dry_run": (
                "AVG calculates the average marks.\n"
                "MAX returns the highest marks.\n"
                "MIN returns the lowest marks."
            ),
            "plan": [
                "What aggregate functions do",
                "COUNT and SUM",
                "AVG, MIN and MAX",
                "Using aggregate functions in queries",
                "Reading query results",
                "A short practice quiz"
            ]
        }
    },
    "Artificial Intelligence": {
        "Introduction to AI": {
            "definition": (
                "Artificial Intelligence is a field that develops "
                "systems capable of tasks such as recognising "
                "patterns, making predictions or processing language."
            ),
            "why": (
                "AI systems can help people analyse data, automate "
                "tasks and support decisions."
            ),
            "example": (
                "Example: A spam filter analyses features of "
                "messages and predicts whether a message is spam."
            ),
            "dry_run": (
                "1. The system receives a message.\n"
                "2. It analyses relevant features.\n"
                "3. A trained model predicts a category.\n"
                "4. The system labels the message based on that prediction."
            ),
            "plan": [
                "What AI means",
                "Examples of AI applications",
                "AI versus traditional programming",
                "Introduction to machine learning",
                "Benefits and limitations",
                "A short practice quiz"
            ]
        },
        "Introduction to Machine Learning": {
            "definition": (
                "Machine learning is a branch of AI in which "
                "algorithms learn patterns from data to make "
                "predictions or decisions."
            ),
            "why": (
                "Learning patterns from examples can help a system "
                "handle new data without writing a separate rule "
                "for every possible case."
            ),
            "example": (
                "A model can learn from labelled examples of "
                "spam and non-spam messages."
            ),
            "dry_run": (
                "1. Collect example messages.\n"
                "2. Label them as spam or non-spam.\n"
                "3. Train a model on the examples.\n"
                "4. Evaluate its predictions on unseen examples."
            ),
            "plan": [
                "What machine learning means",
                "Learning from examples",
                "Training data and labels",
                "Training versus testing",
                "Model evaluation",
                "A short practice quiz"
            ]
        }
    }
}


# =========================================================
# 4. DATABASE
# =========================================================

def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                course TEXT,
                college TEXT,
                academic_year TEXT,
                semester TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS lesson_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                level TEXT NOT NULL,
                explanation TEXT NOT NULL,
                student_response TEXT,
                retry_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS topic_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                level TEXT NOT NULL,
                status TEXT DEFAULT 'Not started',
                last_score INTEGER,
                total_marks INTEGER,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(student_id, subject, topic),
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                level TEXT NOT NULL,
                score INTEGER NOT NULL,
                total_marks INTEGER NOT NULL,
                answers TEXT,
                attempted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
        """)


init_db()


# =========================================================
# 5. PASSWORD SECURITY AND ACCOUNT FUNCTIONS
# =========================================================

def hash_password(password):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        200000
    ).hex()
    return f"{salt}${digest}"


def verify_password(password, stored_hash):
    try:
        salt, saved_digest = stored_hash.split("$", 1)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            200000
        ).hex()
        return secrets.compare_digest(digest, saved_digest)
    except (ValueError, AttributeError):
        return False


def authenticate(email, password):
    with get_connection() as conn:
        student = conn.execute(
            "SELECT * FROM students WHERE email = ?",
            (email.strip().lower(),)
        ).fetchone()

    if student and verify_password(password, student[3]):
        return student
    return None


def create_account(
    full_name, email, password, course,
    college, academic_year, semester
):
    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO students
                (full_name, email, password_hash, course, college,
                 academic_year, semester)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                full_name.strip(),
                email.strip().lower(),
                hash_password(password),
                course.strip(),
                college.strip(),
                academic_year,
                semester
            ))
        return True
    except sqlite3.IntegrityError:
        return False
    except sqlite3.Error as exc:
        raise RuntimeError(f"Database error while creating account: {exc}") from exc


# =========================================================
# 6. STUDENT PROGRESS FUNCTIONS
# =========================================================

def save_lesson(student_id, subject, topic, level,
                explanation, response, retry_count):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO lesson_history
            (student_id, subject, topic, level, explanation,
             student_response, retry_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            student_id, subject, topic, level,
            explanation, response, retry_count
        ))


def update_topic_progress(student_id, subject, topic, level,
                          status, score=None, total=None):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO topic_progress
            (student_id, subject, topic, level, status,
             last_score, total_marks, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(student_id, subject, topic)
            DO UPDATE SET
                level = excluded.level,
                status = excluded.status,
                last_score = COALESCE(excluded.last_score, topic_progress.last_score),
                total_marks = COALESCE(excluded.total_marks, topic_progress.total_marks),
                updated_at = CURRENT_TIMESTAMP
        """, (
            student_id, subject, topic, level,
            status, score, total
        ))



def save_quiz_attempt(student_id, subject, topic, level,
                      score, total, answers):
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO quiz_attempts
            (student_id, subject, topic, level, score,
             total_marks, answers)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            student_id,
            subject,
            topic,
            level,
            score,
            total,
            answers
        ))

        return cursor.lastrowid


def get_student_stats(student_id):
    with get_connection() as conn:

        # 1. Count total quiz attempts
        quiz_count = conn.execute("""
            SELECT COUNT(*)
            FROM quiz_attempts
            WHERE student_id = ?
        """, (student_id,)).fetchone()[0]

        # 2. Fetch all quiz scores
        scores = conn.execute("""
            SELECT score, total_marks
            FROM quiz_attempts
            WHERE student_id = ?
        """, (student_id,)).fetchall()

        # 3. Fetch topic progress
        topics = conn.execute("""
            SELECT subject, topic, status, last_score, total_marks
            FROM topic_progress
            WHERE student_id = ?
            ORDER BY updated_at DESC
        """, (student_id,)).fetchall()

        # 4. Fetch individual quiz attempts
        attempts = conn.execute("""
            SELECT subject, topic, level, score, total_marks
            FROM quiz_attempts
            WHERE student_id = ?
            ORDER BY rowid ASC
        """, (student_id,)).fetchall()

    # 5. Calculate average percentage
    percentage_values = []

    for score, total in scores:
        if total > 0:
            percentage_values.append((score / total) * 100)

    if percentage_values:
        average = round(sum(percentage_values) / len(percentage_values))
    else:
        average = None

    # 6. Return four values
    return quiz_count, average, topics, attempts


# =============================
def ask_gemini_tutor(subject, topic, level, student_question):
    """Generate a student-friendly explanation using Gemini."""

    if "GEMINI_API_KEY" not in st.secrets:
        return "API Key Error: Please check your secrets.toml file."

    try:
        api_key = st.secrets["GEMINI_API_KEY"].strip()

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=30000)
        )

        prompt = f"""
You are Concept Detective AI, a friendly teaching assistant.

Subject: {subject}
Topic: {topic}
Student level: {level}
Student question: {student_question}

Explain the answer in simple language.
Break difficult ideas into small steps.
Give one practical example.
Use headings and bullet points where useful.
Keep the explanation clear and concise.
"""

        for attempt in range(3):
            try:
                print("GEMINI REQUEST STARTED")
                print("GEMINI RESPONSE RECEIVED")
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt
                )

                if response and response.text and response.text.strip():
                    return response.text.strip()

                return "Gemini returned an empty response. Please try again."

            except Exception as e:
                error = str(e).lower()

                temporary_error = any(
                    code in error
                    for code in ["429", "500", "502", "503", "504"]
                )

                if temporary_error and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue

                return f"Gemini Error: {str(e)}"

        return "Gemini is temporarily unavailable. Please try again."

    except Exception as e:
        return f"Unexpected Error: {str(e)}"


def get_lesson(subject, topic):
    subject_lessons = LESSONS.get(subject, {})
    if topic in subject_lessons:
        return subject_lessons[topic]

    return {
        "definition": (
            f"{topic} is an important topic within {subject}. "
            "To understand it well, begin with its purpose, "
            "basic terminology and a simple example."
        ),
        "why": (
            f"Learning {topic} helps build your understanding "
            f"of {subject} and prepares you for related concepts."
        ),
        "example": (
            f"Start by writing down the main idea of {topic}. "
            "Then identify its inputs, process and expected result."
        ),
        "dry_run": (
            "1. Identify the concept being studied.\n"
            "2. Break the example into small steps.\n"
            "3. Explain what happens at each step.\n"
            "4. Check whether the final result follows from those steps."
        ),
        "plan": [
            f"What {topic} means",
            f"Why {topic} is useful in {subject}",
            "Important terms and core ideas",
            "A worked example",
            "A short practice quiz"
        ]
    }


def make_explanation(lesson, retry_count):
    if retry_count == 0:
        return (
            lesson["definition"] + "\n\n"
            + "**Why it matters**\n" + lesson["why"] + "\n\n"
            + "**Example**\n" + lesson["example"] + "\n\n"
            + "**Step-by-step walkthrough**\n" + lesson["dry_run"]
        )

    if retry_count == 1:
        return (
            "**Let's try a simpler explanation.**\n\n"
            + lesson["why"] + "\n\n"
            "**Look at this example carefully:**\n"
            + lesson["example"] + "\n\n"
            "**Follow these steps:**\n"
            + lesson["dry_run"] + "\n\n"
            "Read each step slowly and try explaining it in your own words."
        )

    return (
        "**Let's break it down into smaller parts.**\n\n"
        "**First, understand the main idea:**\n"
        + lesson["definition"] + "\n\n"
        "**Now, connect it to an example:**\n"
        + lesson["example"] + "\n\n"
        "**Trace the example one step at a time:**\n"
        + lesson["dry_run"] + "\n\n"
        "If one part is still confusing, write down that exact doubt "
        "and ask for clarification before attempting the quiz."
    )


# =========================================================
# 8. QUIZ QUESTION BANK
# =========================================================

QUIZ_BANK = {
    "Java": [
        {
            "q": "Which method is the usual entry point of a Java application?",
            "options": [
                "start()", "main()", "runCode()", "begin()"
            ],
            "answer": 1,
            "explain": "The standard entry point is main(String[] args)."
        },
        {
            "q": "Which Java type stores a whole number such as 25?",
            "options": ["int", "boolean", "String", "char"],
            "answer": 0,
            "explain": "int stores integer values."
        },
        {
            "q": "What is the first valid index of a Java array?",
            "options": ["1", "-1", "0", "It has no index"],
            "answer": 2,
            "explain": "Java array indexing starts at zero."
        },
        {
            "q": "Which statement repeats instructions while a condition is true?",
            "options": ["import", "return", "class", "while"],
            "answer": 3,
            "explain": "A while loop repeats while its condition is true."
        },
        {
            "q": "Which operator checks equality between primitive values?",
            "options": ["=", "==", "!=", "++"],
            "answer": 1,
            "explain": "== compares primitive values for equality; = assigns a value."
        }
    ],
    "Python": [
        {
            "q": "Which function displays output in Python?",
            "options": ["echo()", "show()", "print()", "displayText()"],
            "answer": 2,
            "explain": "print() displays output."
        },
        {
            "q": "Which symbol starts a single-line comment in Python?",
            "options": ["//", "#", "<!--", "**"],
            "answer": 1,
            "explain": "A Python single-line comment starts with #."
        },
        {
            "q": "What is the first index of a Python list?",
            "options": ["0", "1", "-1 only", "The list length"],
            "answer": 0,
            "explain": "The first list element is at index 0."
        },
        {
            "q": "Which keyword defines a function in Python?",
            "options": ["function", "fun", "define", "def"],
            "answer": 3,
            "explain": "Python uses def to define a function."
        },
        {
            "q": "Which collection is mutable?",
            "options": ["tuple", "list", "str", "int"],
            "answer": 1,
            "explain": "Lists can be modified after creation."
        }
    ],
    "SQL and DBMS": [
        {
            "q": "Which SQL command retrieves data from a table?",
            "options": ["SELECT", "REMOVE", "CHANGE", "INSERT ONLY"],
            "answer": 0,
            "explain": "SELECT retrieves rows and columns."
        },
        {
            "q": "Which clause filters rows using a condition?",
            "options": ["ORDER BY", "WHERE", "GROUP BY", "CREATE"],
            "answer": 1,
            "explain": "WHERE filters rows according to a condition."
        },
        {
            "q": "Which function counts rows?",
            "options": ["TOTAL()", "NUMBER()", "COUNT()", "ROWSUM()"],
            "answer": 2,
            "explain": "COUNT() counts rows or non-NULL values depending on its argument."
        },
        {
            "q": "Which join returns only rows matching the join condition in both tables?",
            "options": ["CROSS JOIN", "LEFT JOIN", "FULL JOIN", "INNER JOIN"],
            "answer": 3,
            "explain": "INNER JOIN returns matching rows from both sides."
        },
        {
            "q": "Which key uniquely identifies a row in a table?",
            "options": ["Foreign key only", "Primary key", "Duplicate key", "Nullable key"],
            "answer": 1,
            "explain": "A primary key uniquely identifies each row and cannot be NULL."
        }
    ],
    "Artificial Intelligence": [
        {
            "q": "Which field enables systems to learn patterns from data?",
            "options": ["Machine learning", "Word processing", "File compression", "Screen rendering"],
            "answer": 0,
            "explain": "Machine learning develops models from data."
        },
        {
            "q": "What is a feature in a dataset?",
            "options": ["Only the final answer", "A data attribute used by a model", "A computer brand", "A file extension"],
            "answer": 1,
            "explain": "Features are input attributes used for learning or prediction."
        },
        {
            "q": "Why should a model be evaluated on unseen data?",
            "options": ["To increase file size", "To remove all training data", "To assess generalisation", "To rename the model"],
            "answer": 2,
            "explain": "Unseen data helps assess performance beyond the training examples."
        },
        {
            "q": "In supervised learning, training examples commonly include:",
            "options": ["No data", "Only computer names", "Only random predictions", "Input data and target labels"],
            "answer": 3,
            "explain": "Supervised learning uses examples with known target labels."
        },
        {
            "q": "Which is a possible AI application?",
            "options": ["Only opening a folder manually", "Spam classification", "Only changing a filename", "Only switching off a monitor"],
            "answer": 1,
            "explain": "AI models can classify messages as spam or non-spam."
        }
    ]
}


JAVA_TOPIC_QUIZZES = {
    "Introduction to Java": [
        {
            "q": "Who developed Java?",
            "options": ["Microsoft", "James Gosling", "Dennis Ritchie", "Guido van Rossum"],
            "answer": 1,
            "explain": "James Gosling led the development of Java at Sun Microsystems."
        },
        {
            "q": "Which feature allows Java programs to run on different platforms?",
            "options": ["Platform independence", "Manual memory allocation", "Multiple inheritance of classes", "Pointer arithmetic"],
            "answer": 0,
            "explain": "Java bytecode runs on systems that have a compatible JVM."
        },
        {
            "q": "What does JVM stand for?",
            "options": ["Java Variable Method", "Java Verified Machine", "Java Virtual Machine", "Java Visual Model"],
            "answer": 2,
            "explain": "JVM stands for Java Virtual Machine."
        },
        {
            "q": "Which file extension is used for Java source files?",
            "options": [".class", ".java", ".js", ".py"],
            "answer": 1,
            "explain": "Java source code files use the .java extension."
        },
        {
            "q": "Which method is the usual entry point of a Java application?",
            "options": ["start()", "run()", "main()", "execute()"],
            "answer": 2,
            "explain": "The main method is the standard entry point of a Java application."
        }
    ],

    "Variables and Data Types": [
        {
            "q": "Which data type stores whole numbers in Java?",
            "options": ["double", "int", "char", "boolean"],
            "answer": 1,
            "explain": "The int data type stores integer values."
        },
        {
            "q": "Which data type stores a single character?",
            "options": ["String", "int", "char", "float"],
            "answer": 2,
            "explain": "The char data type stores a single character."
        },
        {
            "q": "Which data type stores true or false?",
            "options": ["boolean", "String", "double", "long"],
            "answer": 0,
            "explain": "A boolean variable stores either true or false."
        },
        {
            "q": "Which of these is a valid Java variable declaration?",
            "options": ["int number = 10;", "number int = 10;", "integer number = 10;", "10 = number;"],
            "answer": 0,
            "explain": "The data type comes before the variable name in a declaration."
        },
        {
            "q": "Which data type can store decimal values with double precision?",
            "options": ["char", "boolean", "double", "byte"],
            "answer": 2,
            "explain": "The double type stores floating-point values with double precision."
        }
    ],

    "Operators": [
        {
            "q": "Which operator is used for multiplication?",
            "options": ["+", "*", "/", "%"],
            "answer": 1,
            "explain": "The * operator performs multiplication."
        },
        {
            "q": "What is the result of 10 % 3?",
            "options": ["3", "1", "0", "2"],
            "answer": 1,
            "explain": "The modulo operator returns the remainder. The remainder is 1."
        },
        {
            "q": "Which operator checks equality between two values?",
            "options": ["=", "==", "!=", ">"],
            "answer": 1,
            "explain": "The == operator checks whether two values are equal."
        },
        {
            "q": "Which operator represents logical AND?",
            "options": ["||", "!", "&&", "&="],
            "answer": 2,
            "explain": "The && operator performs logical AND."
        },
        {
            "q": "What is the result of 5 + 2 * 3?",
            "options": ["21", "11", "15", "13"],
            "answer": 1,
            "explain": "Multiplication happens before addition, so 5 + 6 = 11."
        }
    ],

    "If-Else Statements": [
        {
            "q": "Which keyword is used to check a condition?",
            "options": ["for", "if", "class", "import"],
            "answer": 1,
            "explain": "The if keyword executes a block when its condition is true."
        },
        {
            "q": "Which block runs when the if condition is false?",
            "options": ["else", "continue", "break", "return"],
            "answer": 0,
            "explain": "The else block runs when the associated if condition is false."
        },
        {
            "q": "Which keyword checks another condition after an if?",
            "options": ["repeat", "otherwise", "elif", "else if"],
            "answer": 3,
            "explain": "Java uses else if to check another condition."
        },
        {
            "q": "What is the output of int x = 5; if (x > 3) System.out.print(\"Yes\");?",
            "options": ["No", "Yes", "5", "Error"],
            "answer": 1,
            "explain": "Since 5 is greater than 3, the condition is true and Yes is printed."
        },
        {
            "q": "Which operator checks whether two values are not equal?",
            "options": ["==", "=", "!=", ">="],
            "answer": 2,
            "explain": "The != operator checks whether two values are different."
        }
    ],

    "Loops": [
        {
            "q": "Which loop is commonly used when the number of iterations is known?",
            "options": ["for", "if", "switch", "try"],
            "answer": 0,
            "explain": "A for loop is commonly used when the iteration count is known."
        },
        {
            "q": "Which loop checks its condition before executing the body?",
            "options": ["do-while", "while", "Both always execute first", "Neither"],
            "answer": 1,
            "explain": "A while loop checks its condition before each iteration."
        },
        {
            "q": "Which statement immediately exits a loop?",
            "options": ["continue", "skip", "break", "next"],
            "answer": 2,
            "explain": "The break statement terminates the loop."
        },
        {
            "q": "How many times does for(int i = 0; i < 3; i++) run?",
            "options": ["2", "3", "4", "0"],
            "answer": 1,
            "explain": "The loop runs for i = 0, 1, and 2: three iterations."
        },
        {
            "q": "Which loop executes its body at least once?",
            "options": ["for", "while", "do-while", "None of these"],
            "answer": 2,
            "explain": "A do-while loop checks its condition after executing the body."
        }
    ],

    "Arrays": [
        {
            "q": "What is the index of the first element in a Java array?",
            "options": ["1", "-1", "0", "It depends on the array"],
            "answer": 2,
            "explain": "Java arrays use zero-based indexing."
        },
        {
            "q": "Which property gives the number of elements in an array?",
            "options": ["size()", "length", "length()", "count()"],
            "answer": 1,
            "explain": "An array uses the length property to provide its size."
        },
        {
            "q": "What is the last valid index of an array of length 5?",
            "options": ["5", "4", "3", "6"],
            "answer": 1,
            "explain": "The indexes are 0, 1, 2, 3, and 4."
        },
        {
            "q": "Which declaration creates an integer array?",
            "options": ["int arr[];", "array int arr;", "arr int[];", "integer arr;"],
            "answer": 0,
            "explain": "int arr[]; is a valid Java array declaration."
        },
        {
            "q": "What happens when you access an invalid array index?",
            "options": ["It returns zero", "It returns null", "An ArrayIndexOutOfBoundsException occurs", "The array grows automatically"],
            "answer": 2,
            "explain": "Accessing an index outside the array bounds causes an ArrayIndexOutOfBoundsException."
        }
    ],

    "Strings": [
        {
            "q": "Which class represents strings in Java?",
            "options": ["Text", "String", "Character", "Sentence"],
            "answer": 1,
            "explain": "Java uses the String class to represent text."
        },
        {
            "q": "Which method returns the number of characters in a String?",
            "options": ["size()", "length", "length()", "count()"],
            "answer": 2,
            "explain": "The length() method returns the number of characters."
        },
        {
            "q": "Which method compares the contents of two strings?",
            "options": ["==", "equals()", "compare", "same()"],
            "answer": 1,
            "explain": "equals() compares the contents of two strings."
        },
        {
            "q": "Are Java String objects mutable?",
            "options": ["Yes, always", "No, they are immutable", "Only inside loops", "Only when declared final"],
            "answer": 1,
            "explain": "String objects are immutable; operations create new strings rather than changing the original."
        },
        {
            "q": "Which method converts a string to uppercase?",
            "options": ["upper()", "toUpperCase()", "uppercase()", "convertUpper()"],
            "answer": 1,
            "explain": "toUpperCase() returns a string with uppercase characters."
        }
    ],

    "Methods": [
        {
            "q": "What is a method in Java?",
            "options": ["A reusable block of code", "A data type", "An array index", "A package name"],
            "answer": 0,
            "explain": "A method is a block of code designed to perform a task."
        },
        {
            "q": "Which keyword sends a value back from a method?",
            "options": ["break", "return", "continue", "static"],
            "answer": 1,
            "explain": "The return statement sends a value back to the caller."
        },
        {
            "q": "Which return type is used when a method returns no value?",
            "options": ["null", "empty", "void", "int"],
            "answer": 2,
            "explain": "The void keyword indicates that a method does not return a value."
        },
        {
            "q": "What is a parameter?",
            "options": ["A value received by a method", "A loop condition", "An array size", "A class name"],
            "answer": 0,
            "explain": "Parameters are variables listed in a method declaration to receive input."
        },
        {
            "q": "What is method overloading?",
            "options": ["Using multiple classes", "Defining methods with the same name but different parameter lists", "Calling a method repeatedly", "Changing a variable's value"],
            "answer": 1,
            "explain": "Method overloading uses the same method name with different parameter lists."
        }
    ],

    "Object-Oriented Programming": [
        {
            "q": "Which concept combines data and methods into a single unit?",
            "options": ["Encapsulation", "Iteration", "Compilation", "Recursion"],
            "answer": 0,
            "explain": "Encapsulation bundles data and the methods that operate on it."
        },
        {
            "q": "Which keyword is used to inherit a class in Java?",
            "options": ["implements", "extends", "inherits", "super"],
            "answer": 1,
            "explain": "The extends keyword is used for class inheritance."
        },
        {
            "q": "What is an object?",
            "options": ["An instance of a class", "A loop", "A package", "A primitive data type"],
            "answer": 0,
            "explain": "An object is an instance of a class."
        },
        {
            "q": "Which OOP concept allows one interface to have different implementations?",
            "options": ["Encapsulation", "Polymorphism", "Compilation", "Initialization"],
            "answer": 1,
            "explain": "Polymorphism allows the same interface or method call to show different behaviour."
        },
        {
            "q": "Which keyword refers to the current object?",
            "options": ["super", "this", "static", "new"],
            "answer": 1,
            "explain": "The this keyword refers to the current object."
        }
    ]
}


def get_quiz(subject, topic):
    if subject == "Java":
        if topic in JAVA_TOPIC_QUIZZES:
            return JAVA_TOPIC_QUIZZES[topic]
        return QUIZ_BANK["Java"]

    if subject in QUIZ_BANK:
        return QUIZ_BANK[subject]

    return [
        {
            "q": f"What is a useful first step when learning {topic}?",
            "options": [
                "Memorise random words",
                "Ignore definitions",
                "Understand the core concept",
                "Skip all examples"
            ],
            "answer": 2,
            "explain": "Understanding the core concept gives later examples a foundation."
        },
        {
            "q": f"Why are worked examples useful for {topic}?",
            "options": [
                "They show how ideas are applied",
                "They remove the need to understand",
                "They always guarantee full marks",
                "They replace all practice"
            ],
            "answer": 0,
            "explain": "Worked examples show how a concept can be applied step by step."
        },
        {
            "q": "What should you do when a concept is confusing?",
            "options": [
                "Guess every answer",
                "Stop reviewing it forever",
                "Skip every related topic",
                "Ask a specific question and review an example"
            ],
            "answer": 3,
            "explain": "Specific questions and worked examples help locate the source of confusion."
        },
        {
            "q": "What is a useful way to check your understanding?",
            "options": [
                "Read only the title",
                "Explain the idea in your own words",
                "Avoid any practice",
                "Copy the definition without reading it"
            ],
            "answer": 1,
            "explain": "Explaining an idea in your own words is a useful understanding check."
        },
        {
            "q": "What should you do after a low quiz score?",
            "options": [
                "Review mistakes and practise again",
                "Delete your learning history",
                "Assume every concept is mastered",
                "Never attempt another quiz"
            ],
            "answer": 0,
            "explain": "Reviewing mistakes and practising again can help improve understanding."
        }
    ]


# =========================================================
# 9. CUSTOM CSS
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(0,217,255,0.10), transparent 36%),
        radial-gradient(ellipse at 90% 10%, rgba(139,92,246,0.13), transparent 35%),
        #07111f;
    color: #edf6ff;
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f5f9ff !important;
}

p, label, li {
    color: #d4e1f0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b192a, #10172b);
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: #e5f2ff;
}

.hero {
    padding: 36px;
    border-radius: 25px;
    min-height: 280px;
    background:
        linear-gradient(100deg, rgba(5,11,22,0.96),
        rgba(9,24,43,0.87) 48%,
        rgba(39,24,69,0.64)),
        url("https://images.unsplash.com/photo-1521737711867-e3b97375f902?auto=format&fit=crop&w=1800&q=85")
        center/cover no-repeat;
    border: 1px solid rgba(255,255,255,0.13);
    box-shadow: 0 18px 60px rgba(0,0,0,0.24);
    margin-bottom: 22px;
}

.eyebrow {
    color: #63e7ff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(30px, 4vw, 48px);
    font-weight: 700;
    line-height: 1.1;
    margin: 12px 0;
    color: white;
}

.hero-description {
    max-width: 620px;
    font-size: 16px;
    line-height: 1.8;
    color: #d9e8f8;
}

.glass-card {
    background: linear-gradient(145deg, rgba(21,39,61,0.88),
                                rgba(16,24,43,0.91));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 22px;
    margin-bottom: 15px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.12);
}

.metric-card {
    background: linear-gradient(140deg, rgba(16,44,65,0.94),
                                rgba(25,27,57,0.94));
    border: 1px solid rgba(91,225,255,0.18);
    border-radius: 18px;
    padding: 20px;
    min-height: 125px;
}

.metric-label {
    color: #a9c0d8;
    font-size: 13px;
}

.metric-value {
    color: #67e8f9;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 29px;
    font-weight: 700;
    margin-top: 8px;
}

div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(100deg, #00c6e8, #7667ff);
    color: white !important;
    border: 0;
    border-radius: 12px;
    min-height: 44px;
    font-weight: 700;
}

div.stButton > button:hover {
    border: 1px solid #9cefff;
    box-shadow: 0 0 20px rgba(0,217,255,0.20);
    transform: translateY(-1px);
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] > div {
    background-color: #101e31 !important;
    border-color: #3c5675 !important;
    border-radius: 10px;
}

/* Readable text inside inputs and dropdowns */
.stApp input,
.stApp textarea,
div[data-baseweb="select"] span {
    color: #f8faff !important;
    -webkit-text-fill-color: #f8faff !important;
    caret-color: #67e8f9 !important;
}

.stApp input::placeholder,
.stApp textarea::placeholder {
    color: #9fb3cc !important;
    -webkit-text-fill-color: #9fb3cc !important;
}

/* Dropdown menu and options */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
ul[data-baseweb="menu"],
div[role="listbox"],
li[role="option"],
div[role="option"] {
    background-color: #101e31 !important;
    color: #f8faff !important;
}

li[role="option"]:hover,
div[role="option"]:hover {
    background-color: #203b59 !important;
    color: #67e8f9 !important;
}

/* Form labels */
div[data-testid="stRadio"] label,
div[data-testid="stCheckbox"] label,
.stSelectbox label,
.stTextInput label,
.stMultiSelect label {
    color: #f1f7ff !important;
}

/* Selected topic chips */
span[data-baseweb="tag"] {
    background: linear-gradient(120deg, #164e63, #3730a3) !important;
    border: 1px solid #67e8f9 !important;
    border-radius: 8px !important;
}

span[data-baseweb="tag"] * {
    color: white !important;
    fill: white !important;
}

/* Quiz form */
div[data-testid="stForm"] {
    background: rgba(12,27,46,0.70);
    border: 1px solid rgba(103,232,249,0.20);
    border-radius: 18px;
    padding: 22px;
}

div[data-testid="stExpander"] {
    background: rgba(18,34,54,0.72);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 13px;
}

hr {
    border-color: rgba(255,255,255,0.12);
}

.small-muted {
    color: #a8bdd3;
    font-size: 13px;
}

.lesson-box {
    background: rgba(12,29,48,0.88);
    border: 1px solid rgba(0,217,255,0.19);
    border-radius: 16px;
    padding: 22px;
}

.stApp input:focus,
.stApp textarea:focus {
    outline: 2px solid #22d3ee !important;
    outline-offset: 2px;
}

/* Fix text visibility in all input fields */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    color: #ffffff !important;
    background-color: #112033 !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #00d9ff !important;
}

/* Make placeholder text readable */
.stTextInput input::placeholder,
.stTextArea textarea::placeholder,
.stNumberInput input::placeholder {
    color: #aab8cc !important;
    -webkit-text-fill-color: #aab8cc !important;
    opacity: 1 !important;
}

/* Keep labels readable */
.stTextInput label,
.stTextArea label,
.stNumberInput label {
    color: #dbe7f5 !important;
}

/* Chrome autofill fix */
.stApp input:-webkit-autofill,
.stApp input:-webkit-autofill:hover,
.stApp input:-webkit-autofill:focus,
.stApp input:-webkit-autofill:active {
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #00d9ff !important;
    -webkit-box-shadow: 0 0 0 1000px #112033 inset !important;
    box-shadow: 0 0 0 1000px #112033 inset !important;
    transition: background-color 999999s ease-in-out 0s !important;
}

/* Force dark backgrounds and visible text in Streamlit fields */
.stApp div[data-baseweb="input"] input,
.stApp div[data-baseweb="textarea"] textarea {
    background-color: transparent !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Keep the complete input container dark */
.stApp div[data-baseweb="input"] > div {
    background-color: #112033 !important;
}

/* Password field and its surrounding container */
.stApp div[data-testid="stTextInput"] div[data-baseweb="input"] {
    background-color: #112033 !important;
    border-radius: 10px !important;
}

/* Consistent image sizing throughout the app */
div[data-testid="stImage"] img {
    object-fit: cover !important;
    border-radius: 14px !important;
}

/* Dark backgrounds and white text in all form fields */
.stApp input,
.stApp textarea,
.stApp div[data-baseweb="select"] input {
    background-color: #112033 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #67e8f9 !important;
}

/* Dropdown selected value */
.stApp div[data-baseweb="select"] span {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Dropdown menu */
div[data-baseweb="popover"],
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li {
    background-color: #112033 !important;
    color: #ffffff !important;
}

/* Dropdown option hover */
div[role="option"]:hover,
li[role="option"]:hover {
    background-color: #244565 !important;
    color: #67e8f9 !important;
}

/* Radio and checkbox text */
.stApp div[data-testid="stRadio"] label,
.stApp div[data-testid="stCheckbox"] label {
    color: #ffffff !important;
}

/* Form labels */
.stApp .stTextInput label,
.stApp .stSelectbox label,
.stApp .stRadio label,
.stApp .stTextArea label {
    color: #e6f2ff !important;
}

/* Improve bordered subject cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: rgba(103, 232, 249, 0.25) !important;
    border-radius: 18px !important;
}

/* Fix Streamlit top header */
header[data-testid="stHeader"] {
    background: #07111f !important;
    border-bottom: none !important;
}

div[data-testid="stToolbar"] {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 10. SESSION STATE
# =========================================================

defaults = {
    "student": None,
    "page": "My Dashboard",
    "subject": "Java",
    "level": "Beginner",
    "topic": None,
    "learning_plan": [],
    "plan_confirmed": False,
    "lesson_retry": 0,
    "lesson_response": None,
    "lesson_started": False,
    "quiz_submitted": False,
    "quiz_score": None,
    "last_quiz_subject": None,
    "last_quiz_topic": None,
    "login_mode": "Login",
    "custom_subject": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 11. COMMON UI HELPERS
# =========================================================

def hero(title, description, eyebrow="PERSONALISED LEARNING"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def metric(label, value, caption=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="small-muted">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_subject_options():
    return list(SUBJECT_TOPICS.keys()) + ["Other subject"]


def get_topics_for_subject(subject):
    if subject in SUBJECT_TOPICS:
        return SUBJECT_TOPICS[subject]
    return [
        "Introduction to the Subject",
        "Basic Concepts",
        "Important Terminology",
        "Core Principles",
        "Applications and Examples"
    ]


def current_subject():
    return st.session_state.subject


def reset_lesson_state():
    st.session_state.topic = None
    st.session_state.learning_plan = []
    st.session_state.plan_confirmed = False
    st.session_state.lesson_retry = 0
    st.session_state.lesson_response = None
    st.session_state.lesson_started = False
    st.session_state.quiz_submitted = False
    st.session_state.quiz_score = None


# =========================================================
# 12. LOGIN AND REGISTRATION
# =========================================================

def auth_page():
    hero(
        "Learn smarter. Grow stronger.",
        "A personalised learning space that helps you understand "
        "concepts, practise what you learn and track your improvement.",
        "WELCOME TO CONCEPT DETECTIVE AI"
    )

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.image(
            STUDY_IMAGE,
            caption="Your learning journey, organised in one place.",
            use_container_width=True
        )
        st.markdown("""
        <div class="glass-card">
            <h3>Inside your learning workspace</h3>
            <p>✦ Learn multiple subjects at your own pace</p>
            <p>✦ See your learning plan before each lesson</p>
            <p>✦ Ask for another explanation when you need it</p>
            <p>✦ Save quizzes, marks and learning progress</p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        mode = st.radio(
            "Choose an option",
            ["Login", "Create Account"],
            horizontal=True,
            key="auth_mode"
        )

        if mode == "Login":
            st.subheader("Welcome back")
            st.caption("Log in to continue your saved learning journey.")

            with st.form("login_form"):
                email = st.text_input("Email address")
                password = st.text_input("Password", type="password")

                submitted = st.form_submit_button(
                    "Login to my account",
                    use_container_width=True
                )

            if submitted:
                student = authenticate(email, password)

                if student:
                    st.session_state.student = student
                    st.session_state.page = "My Dashboard"
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

        else:
            st.subheader("Create your account")
            st.caption("Your account keeps your learning progress separate.")

            with st.form("register_form"):
                full_name = st.text_input("Full name")
                email = st.text_input("Email address")
                password = st.text_input("Create password", type="password")
                confirm = st.text_input("Confirm password", type="password")

                course = st.selectbox(
                    "Course / Degree",
                    [
                        "B.Tech / B.E.", "BCA", "B.Sc.", "B.Com.",
                        "BBA", "B.A.", "LL.B.", "BBA LL.B.",
                        "B.Com. LL.B.", "MCA", "M.Tech / M.E.",
                        "M.Sc.", "MBA", "LL.M.", "Diploma", "Other"
                    ]
                )

                college = st.text_input(
                    "College Name",
                    placeholder="Enter your college name"
                )

                academic_year = st.selectbox(
                    "Academic year",
                    ["1st Year", "2nd Year", "3rd Year", "4th Year", "5th Year", "Other"]
                )

                semester = st.selectbox(
                    "Semester",
                    [
                        "Semester 1", "Semester 2", "Semester 3",
                        "Semester 4", "Semester 5", "Semester 6",
                        "Semester 7", "Semester 8", "Semester 9",
                        "Semester 10", "Other"
                    ]
                )

                submitted = st.form_submit_button(
                    "Create my account",
                    use_container_width=True
                )

            if submitted:
                if not full_name.strip() or not email.strip() or not password:
                    st.error("Please complete your name, email and password.")
                elif len(password) < 6:
                    st.error("Use a password with at least 6 characters.")
                elif password != confirm:
                    st.error("The passwords do not match.")
                elif "@" not in email or "." not in email:
                    st.error("Please enter a valid email address.")
                else:
                    try:
                        created = create_account(
                            full_name, email, password, course,
                            college, academic_year, semester
                        )
                    except RuntimeError as exc:
                        st.error(str(exc))
                    else:
                        if created:
                            student = authenticate(email, password)

                            if student:
                                st.success("Account created successfully!")
                                st.session_state.student = student
                                st.session_state.page = "My Dashboard"
                                st.rerun()
                            else:
                                st.error(
                                    "Your account was created, but automatic login failed. "
                                    "Please switch to Login and enter your details."
                                )
                        else:
                            st.error(
                                "An account with this email already exists. "
                                "Please log in instead or use another email address."
                            )

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 13. SIDEBAR
# =========================================================

def _sync_navigation():
    st.session_state.page = st.session_state.navigation_radio


def sidebar_navigation():
    student = st.session_state.student

    with st.sidebar:
        st.markdown("## 🔍 Concept Detective")
        st.caption("Your personal learning workspace")
        st.divider()

        st.markdown(f"### Hi, {student[1].split()[0]} 👋")
        st.caption(student[2])

        pages = [
            "My Dashboard",
            "Learn a Subject",
            "My Profile",
            "My Progress",
            "Study Planner"
        ]

        if st.session_state.page in pages:
            st.session_state.navigation_radio = st.session_state.page
        else:
            st.session_state.navigation_radio = "Learn a Subject"

        current_page = st.session_state.page

        st.radio(
            "NAVIGATION",
            pages,
            key="navigation_radio",
            on_change=_sync_navigation
        )

        if current_page not in pages:
            st.caption(f"📍 Current section: {current_page}")
        st.divider()
        st.markdown("**Your learning, your pace.**")
        st.caption("Choose a subject, understand it, practise and improve.")

        if st.button("Log Out", use_container_width=True):
            st.session_state.student = None
            reset_lesson_state()
            st.session_state.page = "My Dashboard"
            st.session_state.navigation_radio = "My Dashboard"
            st.rerun()


# =========================================================
# 14. DASHBOARD
# =========================================================

def dashboard_page():
    student = st.session_state.student
    quiz_count, average, topics, attempts = get_student_stats(student[0])
    hero(
        f"Welcome back, {student[1].split()[0]}!",
        "Your next breakthrough starts with one concept. "
        "Choose a subject, follow your learning plan and build "
        "your understanding step by step.",
        "YOUR PERSONAL LEARNING SPACE"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric("QUIZ ATTEMPTS", quiz_count, "All subjects")
    with c2:
        metric(
            "AVERAGE SCORE",
            f"{average}%" if average is not None else "—",
            "Across completed quizzes"
        )
    with c3:
        completed = sum(1 for row in topics if row[2] == "Completed")
        metric("TOPICS COMPLETED", completed, "Across all subjects")

    st.write("")
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🚀 Continue your learning journey")
        st.write(
            "Pick up where you stopped, explore another subject, "
            "or start a new topic from the beginning."
        )

        if st.button("Explore Subjects →", use_container_width=True):
            st.session_state.page = "Learn a Subject"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Your recent learning")
        if topics:
            for subject, topic, status, score, total in topics[:5]:
                score_text = (
                    f" · Last score: {score}/{total}"
                    if score is not None and total else ""
                )
                st.markdown(
                    f"**{subject}** — {topic}  \n"
                    f"<span class='small-muted'>{status}{score_text}</span>",
                    unsafe_allow_html=True
                )
                st.divider()
        else:
            st.info("Your completed lessons and quiz results will appear here.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.image(
            STUDY_IMAGE,
            caption="Build understanding one concept at a time.",
            use_container_width=True
        )
        st.markdown("""
        <div class="glass-card">
            <h3>How your tutor works</h3>
            <p>01 — Choose a subject and level</p>
            <p>02 — Review your learning plan</p>
            <p>03 — Learn and ask questions</p>
            <p>04 — Take a quiz and track progress</p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# 15. SUBJECT AND TOPIC SELECTION
# =========================================================

def subject_page():
    hero(
        "What would you like to learn?",
        "Choose a subject, explore the visual subject cards, "
        "select your level and build your personalised learning path.",
        "YOUR LEARNING PATH"
    )

    subject_visuals = [
        (
            "Java",
            "https://images.unsplash.com/photo-1515879218367-8466d910aaa4?auto=format&fit=crop&w=600&q=80",
            "Programming fundamentals and OOP"
        ),
        (
            "Python",
            "https://images.unsplash.com/photo-1526379095098-d400fd0bf935?auto=format&fit=crop&w=600&q=80",
            "Readable code and automation"
        ),
        (
            "SQL and DBMS",
            "https://images.unsplash.com/photo-1544383835-bda2bc66a55d?auto=format&fit=crop&w=600&q=80",
            "Databases, queries and joins"
        ),
        (
            "Data Structures and Algorithms",
            "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=600&q=80",
            "Arrays, searching and problem solving"
        ),
        (
            "Artificial Intelligence",
            "https://images.unsplash.com/photo-1677442136019-21780ecad995?auto=format&fit=crop&w=600&q=80",
            "AI and machine learning concepts"
        ),
        (
            "Mathematics",
            "https://images.unsplash.com/photo-1635070041078-e363dbe005cb?auto=format&fit=crop&w=600&q=80",
            "Algebra, probability and statistics"
        ),
        (
            "Operating Systems",
            "https://images.unsplash.com/photo-1558494949-ef010cbdcc31?auto=format&fit=crop&w=600&q=80",
            "Processes, memory and scheduling"
        ),
        (
            "Computer Networks",
            "https://images.unsplash.com/photo-1558494949-ef010cbdcc31?auto=format&fit=crop&w=600&q=80",
            "Networking and communication basics"
        )
    ]

    st.subheader("Explore subjects")
    st.caption("Choose a subject below, or select it from the form.")

    cards = st.columns(4, gap="medium")

    for index, (name, image_url, description) in enumerate(subject_visuals):
        with cards[index % 4]:
            with st.container(border=True):
                st.image(image_url, use_container_width=True)

                st.markdown(
                    f"""
                    <div style="
                        height: 58px;
                        display: flex;
                        align-items: center;
                        font-size: 17px;
                        font-weight: 700;
                        color: #ffffff;
                    ">
                        {name}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div style="
                        height: 48px;
                        color: #b9cce0;
                        font-size: 13px;
                        display: flex;
                        align-items: flex-start;
                    ">
                        {description}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if st.button(
                    "Explore Course →",
                    key=f"choose_subject_card_{index}",
                    use_container_width=True
                ):
                    st.session_state.subject = name
                    st.session_state.subject_page_choice = name
                    st.session_state.level = "Beginner"

                    topics = get_topics_for_subject(name)
                    st.session_state.learning_plan = topics
                    st.session_state.topic = topics[0] if topics else None

                    st.session_state.plan_confirmed = False
                    st.session_state.lesson_started = False
                    st.session_state.lesson_retry = 0
                    st.session_state.lesson_response = None
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score = None

                    st.session_state.page = "Learning Plan"
                    st.rerun()

    st.divider()
    st.subheader("Build your learning path")

    options = get_subject_options()

    current_choice = st.session_state.get(
        "subject_page_choice",
        st.session_state.subject
    )

    if current_choice not in options:
        current_choice = "Other subject"

    with st.container(border=True):
        subject_choice = st.selectbox(
            "1. Choose your subject",
            options,
            index=options.index(current_choice),
            key="subject_page_choice"
        )

        if subject_choice == "Other subject":
            custom = st.text_input(
                "Enter your subject name",
                value=st.session_state.custom_subject,
                placeholder="For example, Economics or Biology",
                key="subject_custom_input"
            )
            subject = custom.strip() if custom.strip() else "Other subject"

            if custom.strip():
                st.session_state.custom_subject = custom.strip()
        else:
            subject = subject_choice
            st.session_state.custom_subject = ""

        level = st.radio(
            "2. Choose your current level",
            [
                "Beginner — start from zero",
                "Intermediate — I know the basics",
                "Advanced — focus on deeper concepts"
            ],
            index=0,
            key="subject_page_level",
            horizontal=False
        )

        level_map = {
            "Beginner — start from zero": "Beginner",
            "Intermediate — I know the basics": "Intermediate",
            "Advanced — focus on deeper concepts": "Advanced"
        }

        selected_level = level_map[level]

        topics_available = get_topics_for_subject(subject)

        selected_topics = st.multiselect(
            "3. Choose topics",
            topics_available,
            default=[],
            key="subject_page_topics"
        )

        complete_path = st.checkbox(
            "I want to follow the complete learning path in order",
            key="subject_page_complete_path"
        )

        st.caption(
            "Tip: Start with one topic if you are a beginner. "
            "You can return later and choose more topics."
        )

        if st.button(
            "Create My Learning Plan →",
            key="create_learning_plan_button",
            use_container_width=True
        ):
            if not subject.strip() or subject == "Other subject":
                st.error("Please enter the name of your subject.")
            else:
                if complete_path:
                    chosen = topics_available
                else:
                    chosen = selected_topics

                if not chosen:
                    st.warning(
                        "Please select at least one topic "
                        "or choose the complete learning path."
                    )
                else:
                    st.session_state.subject = subject
                    st.session_state.level = selected_level
                    st.session_state.learning_plan = chosen
                    st.session_state.topic = chosen[0]
                    st.session_state.plan_confirmed = False
                    st.session_state.lesson_started = False
                    st.session_state.lesson_retry = 0
                    st.session_state.lesson_response = None
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score = None
                    st.session_state.page = "Learning Plan"
                    st.rerun()


# =========================================================
# 16. LEARNING PLAN — SHOWN BEFORE TEACHING
# =========================================================

def learning_plan_page():
    subject = st.session_state.subject
    level = st.session_state.level
    plan = st.session_state.learning_plan

    hero(
        "Your learning plan is ready",
        f"Subject: {subject} · Level: {level}. "
        "Review the roadmap before starting the first lesson.",
        "PLAN BEFORE PRACTICE"
    )

    if not plan:
        st.warning("Choose a subject and topics first.")
        if st.button("Choose a subject"):
            st.session_state.page = "Learn a Subject"
            st.rerun()
        return

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📚 Topics in your learning journey")

    for i, topic in enumerate(plan, start=1):
        lesson = get_lesson(subject, topic)
        with st.expander(f"Step {i}: {topic}", expanded=(i == 1)):
            st.write(lesson["definition"])
            st.markdown("**What we will cover:**")
            for item in lesson["plan"]:
                st.write(f"• {item}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("How each lesson will work")
    st.write("1. Learn the concept with an explanation and example.")
    st.write("2. Review the steps and ask questions if needed.")
    st.write("3. Confirm understanding or request another explanation.")
    st.write("4. Start a quiz when you are ready.")
    st.write("5. Save the score and decide what to learn next.")

    if st.button("Start First Lesson →", use_container_width=True):
        st.session_state.topic = plan[0]
        st.session_state.plan_confirmed = True
        st.session_state.lesson_started = True
        st.session_state.lesson_retry = 0
        st.session_state.lesson_response = None
        st.session_state.quiz_submitted = False
        st.session_state.quiz_score = None

        update_topic_progress(
            st.session_state.student[0],
            subject,
            plan[0],
            level,
            "In progress"
        )

        st.session_state.page = "Lesson"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 17. LESSON AND UNDERSTANDING CHECK
# =========================================================

def lesson_page():
    subject = st.session_state.subject
    topic = st.session_state.topic
    level = st.session_state.level

    if not topic:
        st.warning("Please select a topic first.")
        if st.button("Choose topic"):
            st.session_state.page = "Learn a Subject"
            st.rerun()
        return

    lesson = get_lesson(subject, topic)

    hero(
        topic,
        f"Subject: {subject} · Level: {level}. "
        "Take your time. You can ask for another explanation.",
        "YOUR PERSONAL TUTOR"
    )

    st.progress(
        min(
            (st.session_state.learning_plan.index(topic) + 1)
            / len(st.session_state.learning_plan),
            1.0
        ) if topic in st.session_state.learning_plan else 0.1
    )

    st.markdown('<div class="lesson-box">', unsafe_allow_html=True)
    st.subheader("📖 Let's understand the concept")

    explanation = make_explanation(
        lesson,
        st.session_state.lesson_retry
    )
    st.markdown(explanation)

    save_key = (
        f"lesson_saved_{subject}_{topic}_"
        f"{st.session_state.lesson_retry}"
    )

    if not st.session_state.get(save_key, False):
        save_lesson(
            st.session_state.student[0],
            subject,
            topic,
            level,
            explanation,
            st.session_state.lesson_response or "Lesson displayed",
            st.session_state.lesson_retry
        )
        st.session_state[save_key] = True

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Show previous Gemini answer if it exists ──
    if st.session_state.get("gemini_doubt_topic") == topic and st.session_state.get("gemini_doubt_answer"):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🤖 Gemini AI Tutor's Answer")
        st.markdown(st.session_state.gemini_doubt_answer)
        st.markdown("</div>", unsafe_allow_html=True)
        st.divider()

    # ── Understanding check ──
    st.write("")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("💬 Did you understand this explanation?")

    response = st.radio(
        "Choose the option that best describes your understanding:",
        [
            "Yes, I understood",
            "No, please explain again with another example",
            "I understood partly; I have a doubt"
        ],
        key=f"understanding_{subject}_{topic}_{st.session_state.lesson_retry}"
    )

    doubt = ""

    if response == "I understood partly; I have a doubt":
        doubt = st.text_area(
            "What part is confusing?",
            placeholder="Write your question here...",
            key=f"doubt_{subject}_{topic}_{st.session_state.lesson_retry}"
        )

    if st.button("Continue →", use_container_width=True):
        st.session_state.lesson_response = response

        if response == "Yes, I understood":
            # Clear any old doubt answer
            st.session_state.gemini_doubt_answer = None
            st.session_state.gemini_doubt_topic = None
            st.session_state.page = "Ready for Quiz"
            st.rerun()

        elif response == "No, please explain again with another example":
            st.session_state.lesson_retry += 1
            st.session_state.gemini_doubt_answer = None
            st.session_state.gemini_doubt_topic = None

            st.session_state[
                f"lesson_saved_{subject}_{topic}_{st.session_state.lesson_retry}"
            ] = False

            st.rerun()

        else:
            # "I understood partly; I have a doubt"
            if not doubt.strip():
                st.warning("Please write your doubt so the tutor can help.")
            else:
                with st.spinner("🤖 Gemini is thinking about your doubt..."):
                    answer = ask_gemini_tutor(
                        subject,
                        topic,
                        level,
                        doubt.strip()
                    )

                st.session_state.gemini_doubt_answer = answer
                st.session_state.gemini_doubt_topic = topic
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 18. READINESS BEFORE QUIZ
# =========================================================

def ready_page():
    hero(
        "Ready to practise?",
        f"You have reviewed {st.session_state.topic}. "
        "The quiz will help check what you have understood.",
        "KNOWLEDGE CHECK"
    )

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Before you begin")
    st.write("• The quiz has five questions.")
    st.write("• Each correct answer earns one mark.")
    st.write("• Your score will be saved to your profile.")
    st.write("• Incorrect answers will show explanations.")
    st.write("• Your result helps identify concepts for revision.")

    ready = st.radio(
        "Are you ready to start?",
        ["Yes, start the quiz", "No, I want to review the lesson"],
        key=f"ready_{st.session_state.subject}_{st.session_state.topic}"
    )

    if st.button("Confirm →", use_container_width=True):
        if ready == "Yes, start the quiz":
            st.session_state.quiz_submitted = False
            st.session_state.quiz_score = None
            st.session_state.page = "Topic Quiz"
        else:
            st.session_state.page = "Lesson"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# 19. QUIZ AND MARKS
# =========================================================

def quiz_page():
    subject = st.session_state.subject
    topic = st.session_state.topic
    level = st.session_state.level
    questions = get_quiz(subject, topic)

    hero(
        f"{topic} — Practice Quiz",
        "Answer each question carefully. Your marks and feedback "
        "will be stored in your learning profile.",
        "TEST YOUR UNDERSTANDING"
    )

    if not questions:
        st.warning("No quiz questions are available for this topic.")
        return

    if not st.session_state.quiz_submitted:
        with st.form(f"quiz_form_{subject}_{topic}"):
            answers = []

            for i, question in enumerate(questions, start=1):
                st.markdown(f"### Question {i}")
                st.write(question["q"])

                choice = st.radio(
                    "Select one answer:",
                    question["options"],
                    index=None,
                    key=f"quiz_{subject}_{topic}_{i}",
                    label_visibility="collapsed"
                )

                answers.append(choice)

            submitted = st.form_submit_button(
                "Submit Quiz",
                use_container_width=True
            )

        if submitted:
            if any(answer is None for answer in answers):
                st.error("Please answer all questions before submitting.")
                return

            score = 0
            answer_records = []

            for question, selected in zip(questions, answers):
                correct_answer = question["options"][question["answer"]]
                is_correct = selected == correct_answer

                if is_correct:
                    score += 1

                answer_records.append(
                    f"Question: {question['q']} | "
                    f"Selected: {selected} | "
                    f"Correct: {correct_answer} | "
                    f"Result: {'Correct' if is_correct else 'Incorrect'}"
                )

            answer_text = "\n".join(answer_records)

            save_quiz_attempt(
                st.session_state.student[0],
                subject,
                topic,
                level,
                score,
                len(questions),
                answer_text
            )

            percentage = round((score / len(questions)) * 100)

            status = (
                "Completed"
                if percentage >= 80
                else "Needs revision"
            )

            update_topic_progress(
                st.session_state.student[0],
                subject,
                topic,
                level,
                status,
                score,
                len(questions)
            )

            st.session_state.quiz_submitted = True
            st.session_state.quiz_score = score
            st.session_state.quiz_answers = answers

            st.rerun()

    else:
        score = st.session_state.quiz_score
        total = len(questions)
        percentage = round((score / total) * 100)

        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True
        )

        st.subheader("📊 Your Quiz Result")

        c1, c2, c3 = st.columns(3)

        with c1:
            metric("YOUR SCORE", f"{score}/{total}", "Marks earned")

        with c2:
            metric("PERCENTAGE", f"{percentage}%", "Quiz performance")

        with c3:
            metric(
                "RESULT",
                "Strong" if percentage >= 80 else "Keep practising",
                "Based on this attempt"
            )

        if percentage >= 80:
            st.success(
                "Great work! You performed well on this assessment. "
                "Try a new topic or attempt a harder assessment."
            )
        else:
            st.warning(
                "You have some concepts to strengthen. Review the "
                "feedback below and try another explanation or quiz."
            )

        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Answer review")

        answers = st.session_state.get("quiz_answers", [])

        for i, (question, selected) in enumerate(
            zip(questions, answers),
            start=1
        ):
            correct_answer = question["options"][question["answer"]]

            with st.expander(f"Question {i}: {question['q']}"):
                st.write(f"**Your answer:** {selected}")
                st.write(f"**Expected answer:** {correct_answer}")
                st.write(question["explain"])

                if selected == correct_answer:
                    st.success("You answered this correctly.")
                else:
                    st.error("Review this concept before trying again.")

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Retry Quiz", use_container_width=True):
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score = None
                st.session_state.quiz_answers = []
                st.rerun()

        with col2:
            if st.button("Continue Learning", use_container_width=True):
                plan = st.session_state.learning_plan

                try:
                    current_index = plan.index(topic)
                except ValueError:
                    current_index = -1

                if current_index >= 0 and current_index + 1 < len(plan):
                    st.session_state.topic = plan[current_index + 1]
                    st.session_state.lesson_retry = 0
                    st.session_state.lesson_response = None
                    st.session_state.lesson_started = True
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score = None
                    st.session_state.quiz_answers = []

                    update_topic_progress(
                        st.session_state.student[0],
                        subject,
                        st.session_state.topic,
                        level,
                        "In progress"
                    )

                    st.session_state.page = "Lesson"

                else:
                    st.session_state.page = "My Progress"

                st.rerun()


# =========================================================
# 20. STUDENT PROFILE
# =========================================================

def profile_page():
    student = st.session_state.student

    hero(
        "My Profile",
        "Your account information and learning identity.",
        "STUDENT ACCOUNT"
    )

    profile_left, profile_right = st.columns([1, 2])

    with profile_left:
        st.image(
            "https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=500&q=85",
            width=180
        )

    with profile_right:
        st.markdown(
            f"""
            <div style="padding-top: 20px;">
                <div style="font-size: 27px; font-weight: 700; color: white;">
                    {student[1]}
                </div>
                <div style="font-size: 15px; color: #67e8f9; margin-top: 8px;">
                    {student[4] or "Student"}
                </div>
                <div style="font-size: 14px; color: #b9cce0; margin-top: 6px;">
                    {student[2]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Personal and academic details")

    details = [
        ("Full name", student[1]),
        ("Email address", student[2]),
        ("Course / Degree", student[4] or "Not provided"),
        ("College name", student[5] or "Not provided"),
        ("Academic year", student[6] or "Not provided"),
        ("Semester", student[7] or "Not provided"),
        ("Account created", student[8] or "Not available")
    ]

    for label, value in details:
        left, right = st.columns([1, 2])
        left.markdown(f"**{label}**")
        right.write(value)
        st.divider()

    st.markdown("</div>", unsafe_allow_html=True)

    quiz_count, average, topics, attempts = get_student_stats(student[0])
    c1, c2 = st.columns(2)

    with c1:
        metric("TOTAL QUIZZES", quiz_count, "Across all subjects")

    with c2:
        metric(
            "AVERAGE SCORE",
            f"{average}%" if average is not None else "—",
            "Saved assessment results"
        )


# =========================================================
# 21. PROGRESS DASHBOARD
# =========================================================

def progress_page():
    student = st.session_state.student
    quiz_count, average, topics, attempts = get_student_stats(student[0])

    hero(
        "Your Learning Progress",
        "Review subject-wise and topic-wise progress, quiz marks "
        "and concepts that may need more practice.",
        "PROGRESS AND IMPROVEMENT"
    )

    c1, c2 = st.columns(2)
    with c1:
        metric("QUIZ ATTEMPTS", quiz_count, "Saved attempts")
    with c2:
        metric(
            "AVERAGE SCORE",
            f"{average}%" if average is not None else "—",
            "Across all saved attempts"
        )

    st.subheader("Topic performance")

    if not topics:
        st.info("Complete a lesson and quiz to see progress here.")
    else:
        for subject, topic, status, score, total in topics:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"### {subject}")
            st.write(f"**Topic:** {topic}")
            st.write(f"**Status:** {status}")

            if score is not None and total:
                percentage = score / total
                st.write(f"**Latest score:** {score}/{total}")
                st.progress(percentage)

                if percentage < 0.8:
                    st.warning("Recommendation: revise this topic and practise again.")
                else:
                    st.success("Good performance on the latest attempt.")
            else:
                st.caption("No quiz score recorded yet.")

            st.markdown("</div>", unsafe_allow_html=True)

    with get_connection() as conn:
        attempts = conn.execute("""
            SELECT subject, topic, score, total_marks, attempted_at
            FROM quiz_attempts
            WHERE student_id = ?
            ORDER BY attempted_at DESC
            LIMIT 20
        """, (student[0],)).fetchall()

    st.subheader("Recent quiz history")

    if attempts:
        for subject, topic, score, total, date in attempts:
            st.markdown(
                f"**{subject} — {topic}**  \n"
                f"Marks: {score}/{total} · Attempted: {date}"
            )
            st.divider()
    else:
        st.info("Your quiz history will appear here after your first quiz.")


# =========================================================
# 22. STUDY PLANNER
# =========================================================

def planner_page():
    student = st.session_state.student
    _, _, topics, _ = get_student_stats(student[0])

    hero(
        "Your Study Planner",
        "Use your saved results to decide what to revise next.",
        "PERSONALISED REVISION"
    )

    if not topics:
        st.info(
            "Your planner will become more useful after you complete "
            "lessons and quizzes. Start by learning one topic."
        )
        if st.button("Start Learning"):
            st.session_state.page = "Learn a Subject"
            st.rerun()
        return

    weak_topics = [
        row for row in topics
        if row[3] is not None and row[4]
        and (row[3] / row[4]) < 0.8
    ]

    if weak_topics:
        st.subheader("🎯 Recommended revision")
        for subject, topic, status, score, total in weak_topics:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"### {topic}")
            st.write(f"**Subject:** {subject}")
            st.write(f"**Latest score:** {score}/{total}")
            st.write(
                "Suggested plan: revisit the explanation, work through "
                "a new example and attempt the quiz again."
            )

            if st.button(
                f"Revise {topic}",
                key=f"revise_{subject}_{topic}",
                use_container_width=True
            ):
                st.session_state.subject = subject
                st.session_state.topic = topic
                st.session_state.level = "Beginner"
                st.session_state.lesson_retry = 0
                st.session_state.lesson_response = None
                st.session_state.lesson_started = True
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score = None
                st.session_state.learning_plan = [topic]
                st.session_state.page = "Lesson"
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.success(
            "No saved topic currently falls below the revision threshold. "
            "Keep practising and attempt more advanced questions when ready."
        )

    st.subheader("Learning routine")
    st.markdown("""
    <div class="glass-card">
        <p><b>Session 1:</b> Learn one new concept and review its example.</p>
        <p><b>Session 2:</b> Practise questions and explain the concept in your own words.</p>
        <p><b>Session 3:</b> Revisit mistakes and retake a quiz.</p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 23. PAGE ROUTER
# =========================================================

if st.session_state.student is None:
    auth_page()
else:
    sidebar_navigation()

    page = st.session_state.page

    if page == "My Dashboard":
        dashboard_page()
    elif page == "Learn a Subject":
        subject_page()
    elif page == "Learning Plan":
        learning_plan_page()
    elif page == "Lesson":
        lesson_page()
    elif page == "Ready for Quiz":
        ready_page()
    elif page == "Topic Quiz":
        quiz_page()
    elif page == "My Profile":
        profile_page()
    elif page == "My Progress":
        progress_page()
    elif page == "Study Planner":
        planner_page()
    else:
        dashboard_page()