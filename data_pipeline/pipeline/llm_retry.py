"""เรียก LLM พร้อม retry และ fallback (แก้ปัญหา 503 UNAVAILABLE ที่เจอในแอป)"""
import time


class LLMUnavailable(RuntimeError):
    pass


def is_transient(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(k in text for k in ("503", "unavailable", "429", "rate", "timeout", "overloaded"))


def call_with_fallback(providers, prompt, retries=3, base_delay=1.5, sleep=time.sleep):
    """providers = [("gemini", fn), ("local", fn)]  โดย fn(prompt) -> str
    ลองตัวแรกซ้ำแบบ backoff ถ้ายังล้มค่อยไปตัวถัดไป คืน (ชื่อ provider, ข้อความ)"""
    errors = []
    for name, fn in providers:
        for attempt in range(1, retries + 1):
            try:
                return name, fn(prompt)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                if not is_transient(exc):
                    break                       # error ถาวร ไม่ต้อง retry
                if attempt < retries:
                    sleep(base_delay * 2 ** (attempt - 1))
    raise LLMUnavailable("ตอนนี้ระบบ AI ไม่พร้อมใช้งาน กรุณาลองใหม่อีกครั้ง | " + " | ".join(errors[-3:]))
