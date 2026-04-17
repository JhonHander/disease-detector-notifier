import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
DESTINATION_PHONE_NUMBER = os.getenv("DESTINATION_PHONE_NUMBER")


def send_sms_diagnosis(
    diagnosis: str, confidence: float, recipient: str = None
) -> dict:
    if not all([ACCOUNT_SID, AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
        return {"success": False, "error": "Twilio credentials not configured"}

    phone_number = recipient or DESTINATION_PHONE_NUMBER
    if not phone_number:
        return {"success": False, "error": "No destination phone number configured"}

    client = Client(ACCOUNT_SID, AUTH_TOKEN)

    message_body = (
        f"Disease Detector Diagnosis:\n"
        f"Result: {diagnosis}\n"
        f"Confidence: {confidence:.2%}"
    )

    try:
        message = client.messages.create(
            to=phone_number, from_=TWILIO_PHONE_NUMBER, body=message_body
        )
        return {"success": True, "message_sid": message.sid}
    except Exception as e:
        return {"success": False, "error": str(e)}
