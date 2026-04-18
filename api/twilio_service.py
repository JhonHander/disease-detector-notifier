import os
import re
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
DESTINATION_PHONE_NUMBER = os.getenv("DESTINATION_PHONE_NUMBER")
TWILIO_MESSAGING_SERVICE_SID = os.getenv("TWILIO_MESSAGING_SERVICE_SID")

PHONE_PATTERN = re.compile(r"^\+[1-9]\d{7,14}$")


def _is_valid_e164(phone_number: str) -> bool:
    if not phone_number:
        return False
    return bool(PHONE_PATTERN.match(phone_number.strip()))


def get_twilio_config_status() -> str:
    lines = []
    lines.append("Credentials:")
    lines.append(
        f"- TWILIO_ACCOUNT_SID: {'OK' if ACCOUNT_SID else 'MISSING'}"
    )
    lines.append(
        f"- TWILIO_AUTH_TOKEN: {'OK' if AUTH_TOKEN else 'MISSING'}"
    )
    lines.append(
        f"- TWILIO_PHONE_NUMBER: {'OK' if TWILIO_PHONE_NUMBER else 'MISSING'}"
    )
    lines.append(
        "- TWILIO_MESSAGING_SERVICE_SID: "
        f"{'OK' if TWILIO_MESSAGING_SERVICE_SID else 'NOT SET (optional)'}"
    )
    lines.append(
        f"- DESTINATION_PHONE_NUMBER: {'OK' if DESTINATION_PHONE_NUMBER else 'NOT SET'}"
    )

    if TWILIO_PHONE_NUMBER and not _is_valid_e164(TWILIO_PHONE_NUMBER):
        lines.append("Warning: TWILIO_PHONE_NUMBER is not valid E.164 format.")

    if DESTINATION_PHONE_NUMBER and not _is_valid_e164(DESTINATION_PHONE_NUMBER):
        lines.append("Warning: DESTINATION_PHONE_NUMBER is not valid E.164 format.")

    lines.append("")
    lines.append("Tip: Trial accounts can only send SMS to verified destination numbers.")
    return "\n".join(lines)


def send_sms_diagnosis(
    diagnosis: str, confidence: float, recipient: str = None
) -> dict:
    if not all([ACCOUNT_SID, AUTH_TOKEN]):
        return {"success": False, "error": "Twilio credentials not configured"}

    if not TWILIO_PHONE_NUMBER and not TWILIO_MESSAGING_SERVICE_SID:
        return {
            "success": False,
            "error": "Missing sender configuration: set TWILIO_PHONE_NUMBER or TWILIO_MESSAGING_SERVICE_SID",
        }

    if TWILIO_PHONE_NUMBER and not _is_valid_e164(TWILIO_PHONE_NUMBER):
        return {
            "success": False,
            "error": "TWILIO_PHONE_NUMBER must be E.164 format, e.g. +15017122661",
        }

    phone_number = recipient or DESTINATION_PHONE_NUMBER
    if not phone_number:
        return {"success": False, "error": "No destination phone number configured"}

    if not _is_valid_e164(phone_number):
        return {
            "success": False,
            "error": "Destination phone number must be E.164 format, e.g. +573001234567",
        }

    client = Client(ACCOUNT_SID, AUTH_TOKEN)

    message_body = (
        f"Disease Detector Diagnosis:\n"
        f"Result: {diagnosis}\n"
        f"Confidence: {confidence:.2%}"
    )

    try:
        payload = {
            "to": phone_number,
            "body": message_body,
        }

        if TWILIO_MESSAGING_SERVICE_SID:
            payload["messaging_service_sid"] = TWILIO_MESSAGING_SERVICE_SID
        else:
            payload["from_"] = TWILIO_PHONE_NUMBER

        message = client.messages.create(**payload)
        return {
            "success": True,
            "message_sid": message.sid,
            "status": message.status,
        }
    except TwilioRestException as e:
        return {
            "success": False,
            "error": f"Twilio error {e.code}: {e.msg}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
