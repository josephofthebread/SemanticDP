from hashlib import sha256
from pathlib import Path
from subprocess import check_output
from typing import Any

ROOT = Path(__file__).resolve().parent

MANIFEST_NAME = "_manifest.json"
MANIFEST_PATH = ROOT / MANIFEST_NAME

Row = dict[str, Any]
Span = dict[str, Any]


EXTRACTABLE_LABELS = {
  "first_name": "patient's first name",
  "last_name": "patient's last name",
  "date_of_birth": "date of birth",
  "medical_record_number": "medical record number",
  "health_plan_beneficiary_number": "health plan beneficiary number",
  "phone_number": "phone number",
  "email": "email address",
  "date": "date",
  "time": "time",
  "age": "age",
  "street_address": "street address",
  "employee_id": "employee ID",
  "certificate_license_number": "certificate or license number",
  "account_number": "account number",
  "customer_id": "customer ID",
}


def sha256_file(path: Path) -> str:
  digest = sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
      digest.update(chunk)
  return digest.hexdigest()


def git(*command: str) -> str:
  return check_output(["git", *command], cwd=ROOT, text=True).strip()


def build_version(script: Path) -> dict[str, Any]:
  return {
    "script_sha256": sha256_file(script),
    "git_commit": git("rev-parse", "HEAD"),
    "git_dirty": bool(git("status", "--porcelain")),
  }
