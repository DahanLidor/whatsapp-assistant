#!/usr/bin/env bash
# Usage (on the VPS, as root, from inside the project folder):
#   bash setup.sh bot.example.com
set -euo pipefail

DOMAIN="${1:?usage: bash setup.sh <domain>}"
APP_DIR="/opt/whatsapp-assistant"

echo "==> Installing packages"
apt update
apt install -y python3-venv python3-pip caddy ufw

echo "==> Firewall"
ufw allow OpenSSH
ufw allow 80
ufw allow 443
ufw --force enable

echo "==> Copying app to $APP_DIR"
mkdir -p "$APP_DIR"
cp -r ./* ./.env "$APP_DIR"/
id -u assistant &>/dev/null || useradd --system --home "$APP_DIR" assistant
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"
chown -R assistant:assistant "$APP_DIR"
chmod 600 "$APP_DIR/.env"

echo "==> systemd service"
cat > /etc/systemd/system/assistant.service <<EOF
[Unit]
Description=WhatsApp Assistant
After=network.target

[Service]
User=assistant
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now assistant

echo "==> Caddy (automatic HTTPS)"
cat > /etc/caddy/Caddyfile <<EOF
$DOMAIN {
    reverse_proxy 127.0.0.1:8000
}
EOF
systemctl restart caddy

echo "==> Done. Webhook URL: https://$DOMAIN/webhook"
echo "Logs: journalctl -u assistant -f"
