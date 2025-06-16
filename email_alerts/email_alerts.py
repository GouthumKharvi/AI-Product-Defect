
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_alert(model_name, label, confidence):
    sender_email = "samplekharvi@gmail.com"
    receiver_email = "gouthumkharvi1899@gmail.com"
    password = "eqyc xxxy wsvq pvtu"

    subject = f"Alert: Low Confidence Prediction in {model_name}"
    body = f"Prediction: {label}\nConfidence: {confidence:.2f}"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("Email alert sent successfully.")
    except Exception as e:
        print(f"Failed to send email alert: {e}")
