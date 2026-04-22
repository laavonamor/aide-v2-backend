from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import random
import string
from datetime import datetime
from dotenv import load_dotenv

from langchain_gigachat.chat_models import GigaChat
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from supabase import create_client

load_dotenv()

app = FastAPI()

# ─── CORS — разрешаем запросы с Vercel фронтенда ────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # потом ограничишь своим доменом
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Клиенты ─────────────────────────────────────────────────────────────────
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"]
)

giga = GigaChat(
    credentials=os.environ["GIGACHAT_AUTH_KEY"],
    model="GigaChat-Pro",
    verify_ssl_certs=False,
    scope="GIGACHAT_API_PERS",
    profanity_check=False,
)

ADMIN_CODE = os.environ.get("ADMIN_SECRET_CODE", "AIDE_ADMIN")
TODAY = datetime.now().strftime("%Y-%m-%d")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def gen_code():
    return "AIDE-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=4))

def build_system_prompt(profile: dict, org_name: str) -> str:
    LABELS = {
        "name": "Имя и должность",
        "allergies": "Аллергии",
        "diet": "Питание",
        "hotels": "Отели",
        "flights": "Перелёты",
        "schedule": "Расписание",
        "contacts": "Контакты",
        "notes": "Заметки",
    }
    lines = [
        f"• {LABELS[k]}: {profile[k]}"
        for k in LABELS
        if profile.get(k)
    ]
    profile_text = "\n".join(lines) if lines else "⚠️ Профиль не заполнен — напомни пользователю его заполнить."

    return f"""Ты — ИИ-ассистент для персонального помощника руководителя.
Организация: {org_name or '—'}
Сегодня: {TODAY}

ПРОФИЛЬ РУКОВОДИТЕЛЯ:
{profile_text}

ПРАВИЛА:
• Всегда учитывай профиль при каждом ответе
• Аллергии — критично учитывать при любых рекомендациях еды, отелей, ресторанов
• Отвечай по-русски, кратко и конкретно
• Давай реальные варианты со ссылками если возможно

КАЛЕНДАРЬ: Когда просят добавить встречу/событие/звонок — добавь в самый конец ответа блок ТОЧНО в таком формате:
[СОБЫТИЕ: название="...", дата="YYYY-MM-DD", время="HH:MM", длительность=60, описание="..."]
Дата относительная ("завтра", "в пятницу") — вычисли от сегодняшней даты {TODAY}."""

# ─── Модели запросов ──────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    code: str

class ChatRequest(BaseModel):
    orgCode: str
    orgName: str
    message: str

class ProfileRequest(BaseModel):
    code: str
    profile: dict

class AdminRequest(BaseModel):
    secret: str
    action: Optional[str] = None
    name: Optional[str] = None
    orgId: Optional[str] = None

# ─── AUTH ─────────────────────────────────────────────────────────────────────
@app.post("/api/login")
async def login(req: LoginRequest):
    code = req.code.strip().upper()

    if code == ADMIN_CODE:
        return {"role": "admin"}

    result = supabase.table("organizations")\
        .select("id, name, code")\
        .eq("code", code)\
        .eq("active", True)\
        .execute()

    if not result.data:
        raise HTTPException(status_code=403, detail="Неверный или отключённый код доступа")

    org = result.data[0]
    return {"role": "assistant", "orgCode": org["code"], "orgName": org["name"]}

# ─── CHAT ─────────────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(req: ChatRequest):
    # Найти организацию
    org_result = supabase.table("organizations")\
        .select("id")\
        .eq("code", req.orgCode)\
        .eq("active", True)\
        .execute()

    if not org_result.data:
        raise HTTPException(status_code=403, detail="Организация не найдена")

    org_id = org_result.data[0]["id"]

    # Загрузить профиль
    profile_result = supabase.table("profiles")\
        .select("*")\
        .eq("org_id", org_id)\
        .execute()

    profile = profile_result.data[0] if profile_result.data else {}

    # Загрузить историю (последние 20)
    history_result = supabase.table("messages")\
        .select("role, content")\
        .eq("org_id", org_id)\
        .order("created_at", desc=False)\
        .limit(20)\
        .execute()

    # Сохранить сообщение пользователя
    supabase.table("messages").insert({
        "org_id": org_id,
        "role": "user",
        "content": req.message
    }).execute()

    # Собрать историю для LangChain
    messages = [SystemMessage(content=build_system_prompt(profile, req.orgName))]

    for msg in (history_result.data or []):
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=req.message))

    # Вызов GigaChat через LangChain
    try:
        response = giga.invoke(messages)
        reply = response.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Сохранить ответ
    supabase.table("messages").insert({
        "org_id": org_id,
        "role": "assistant",
        "content": reply
    }).execute()

    return {"reply": reply}

# ─── PROFILE ──────────────────────────────────────────────────────────────────
@app.get("/api/profile")
async def get_profile(code: str):
    org = supabase.table("organizations").select("id").eq("code", code).execute()
    if not org.data:
        return {"profile": {}}

    org_id = org.data[0]["id"]
    profile = supabase.table("profiles").select("*").eq("org_id", org_id).execute()
    return {"profile": profile.data[0] if profile.data else {}}

@app.post("/api/profile")
async def save_profile(req: ProfileRequest):
    org = supabase.table("organizations").select("id").eq("code", req.code).execute()
    if not org.data:
        raise HTTPException(status_code=404, detail="Не найдено")

    org_id = org.data[0]["id"]
    data = {**req.profile, "org_id": org_id, "updated_at": datetime.now().isoformat()}

    # upsert — создаст если нет, обновит если есть
    supabase.table("profiles").upsert(data, on_conflict="org_id").execute()
    return {"ok": True}

# ─── HISTORY ──────────────────────────────────────────────────────────────────
@app.get("/api/history")
async def get_history(code: str):
    org = supabase.table("organizations").select("id").eq("code", code).eq("active", True).execute()
    if not org.data:
        return {"messages": []}

    org_id = org.data[0]["id"]
    msgs = supabase.table("messages")\
        .select("id, role, content, created_at")\
        .eq("org_id", org_id)\
        .order("created_at", desc=False)\
        .limit(50)\
        .execute()

    return {"messages": msgs.data or []}

@app.delete("/api/history")
async def clear_history(code: str):
    org = supabase.table("organizations").select("id").eq("code", code).execute()
    if not org.data:
        raise HTTPException(status_code=404, detail="Не найдено")

    org_id = org.data[0]["id"]
    supabase.table("messages").delete().eq("org_id", org_id).execute()
    return {"ok": True}

# ─── ADMIN ────────────────────────────────────────────────────────────────────
@app.get("/api/admin")
async def admin_get(secret: str, org_id: Optional[str] = None):
    if secret != ADMIN_CODE:
        raise HTTPException(status_code=401, detail="Нет доступа")

    if org_id:
        profile = supabase.table("profiles").select("*").eq("org_id", org_id).execute()
        msgs = supabase.table("messages")\
            .select("role, content, created_at")\
            .eq("org_id", org_id)\
            .order("created_at", desc=True)\
            .limit(10)\
            .execute()
        return {"profile": profile.data[0] if profile.data else {}, "messages": msgs.data or []}

    orgs = supabase.table("organizations")\
        .select("id, name, code, active, created_at")\
        .order("created_at", desc=True)\
        .execute()

    result = []
    for org in (orgs.data or []):
        count = supabase.table("messages")\
            .select("id", count="exact")\
            .eq("org_id", org["id"])\
            .execute()
        profile = supabase.table("profiles")\
            .select("name")\
            .eq("org_id", org["id"])\
            .execute()
        result.append({
            **org,
            "msg_count": count.count or 0,
            "profile_filled": bool(profile.data and profile.data[0].get("name"))
        })

    return {"orgs": result}

@app.post("/api/admin")
async def admin_post(req: AdminRequest):
    if req.secret != ADMIN_CODE:
        raise HTTPException(status_code=401, detail="Нет доступа")

    if req.action == "create":
        code = gen_code()
        org = supabase.table("organizations")\
            .insert({"name": req.name, "code": code})\
            .execute()
        return {"org": org.data[0]}

    if req.action == "toggle":
        org = supabase.table("organizations").select("active").eq("id", req.orgId).execute()
        new_status = not org.data[0]["active"]
        supabase.table("organizations").update({"active": new_status}).eq("id", req.orgId).execute()
        return {"ok": True}

    if req.action == "delete":
        supabase.table("organizations").delete().eq("id", req.orgId).execute()
        return {"ok": True}

    raise HTTPException(status_code=400, detail="Неизвестное действие")

# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "service": "AIDE v2 backend"}
