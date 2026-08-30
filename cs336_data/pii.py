import re

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_PATTERN = re.compile(
    r"(?<![\w+])(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]*\d{3}[ .-]*\d{4}(?!\w)"
)
IP_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IP_PATTERN = re.compile(rf"(?<![\w.]){IP_OCTET}(?:\.{IP_OCTET}){{3}}(?!\w|\.\d)")


def mask_emails(text: str) -> tuple[str, int]:
    return EMAIL_PATTERN.subn("|||EMAIL_ADDRESS|||", text)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    return PHONE_PATTERN.subn("|||PHONE_NUMBER|||", text)


def mask_ips(text: str) -> tuple[str, int]:
    return IP_PATTERN.subn("|||IP_ADDRESS|||", text)
