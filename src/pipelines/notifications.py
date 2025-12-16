"""
================================================================================
NOTIFICATION SYSTEM
================================================================================
Email notifications for pipeline events

Sends to: ba8616127@gmail.com

Author: Bilal Ahmad Sheikh
Date: December 2024
================================================================================
"""

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Optional

from .pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


class NotificationManager:
    """Email notification manager for pipeline events (OPTIONAL)"""
    
    def __init__(self):
        self.config = PipelineConfig
        self.smtp_server = self.config.SMTP_SERVER
        self.smtp_port = self.config.SMTP_PORT
        self.sender_email = self.config.SENDER_EMAIL
        self.sender_password = self.config.SENDER_PASSWORD
        self.recipient_email = self.config.NOTIFICATION_EMAIL
        self.enabled = self.config.ENABLE_NOTIFICATIONS and bool(self.sender_password)
        
        if not self.enabled:
            logger.info("📧 Email notifications DISABLED (set ENABLE_NOTIFICATIONS=true in .env to enable)")
    
    def _send_email(self, subject: str, body: str, html: bool = True) -> bool:
        """
        Send email notification
        
        Args:
            subject: Email subject
            body: Email body (HTML or plain text)
            html: Whether body is HTML
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug("Notifications disabled, skipping email")
            return True  # Return True so pipeline continues without error
        
        if not self.sender_password:
            logger.debug("SENDER_PASSWORD not set, skipping email")
            return True  # Return True so pipeline continues without error
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            
            if html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(
                    self.sender_email,
                    self.recipient_email,
                    msg.as_string()
                )
            
            logger.info(f"✅ Email sent: {subject}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to send email: {str(e)}")
            return False
    
    def notify_pipeline_start(self, pipeline_name: str):
        """Send notification when pipeline starts"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        subject = f"🚀 {pipeline_name} - Started"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #0099ff;">🚀 Pipeline Started</h2>
            <p><strong>Pipeline:</strong> {pipeline_name}</p>
            <p><strong>Started At:</strong> {timestamp}</p>
            <hr>
            <p style="color: #666; font-size: 12px;">
                PPP-Q ML Pipeline - Automated Notification
            </p>
        </body>
        </html>
        """
        
        self._send_email(subject, body)
        logger.info(f"Pipeline started: {pipeline_name}")
    
    def notify_pipeline_success(self, pipeline_name: str, details: Dict):
        """Send notification when pipeline completes successfully"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Format details as HTML table
        details_html = ''.join([
            f"<tr><td style='padding: 8px; border: 1px solid #ddd;'><strong>{k}</strong></td>"
            f"<td style='padding: 8px; border: 1px solid #ddd;'>{v}</td></tr>"
            for k, v in details.items()
        ])
        
        subject = f"✅ {pipeline_name} - Success"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #36a64f;">✅ Pipeline Completed Successfully</h2>
            <p><strong>Pipeline:</strong> {pipeline_name}</p>
            <p><strong>Completed At:</strong> {timestamp}</p>
            
            <h3>Results Summary</h3>
            <table style="border-collapse: collapse; width: 100%; margin-top: 15px;">
                <tbody>
                    {details_html}
                </tbody>
            </table>
            
            <hr style="margin-top: 30px;">
            <p style="color: #666; font-size: 12px;">
                PPP-Q ML Pipeline - Automated Notification
            </p>
        </body>
        </html>
        """
        
        self._send_email(subject, body)
        logger.info(f"Pipeline success notification sent: {pipeline_name}")
    
    def notify_pipeline_failure(self, pipeline_name: str, error: str, details: Optional[Dict] = None):
        """Send notification when pipeline fails"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        details_html = ""
        if details:
            details_html = ''.join([
                f"<tr><td style='padding: 8px; border: 1px solid #ddd;'><strong>{k}</strong></td>"
                f"<td style='padding: 8px; border: 1px solid #ddd;'>{v}</td></tr>"
                for k, v in details.items()
            ])
            details_html = f"""
            <h3>Details</h3>
            <table style="border-collapse: collapse; width: 100%; margin-top: 15px;">
                <tbody>
                    {details_html}
                </tbody>
            </table>
            """
        
        subject = f"❌ {pipeline_name} - Failed"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #ff0000;">❌ Pipeline Failed</h2>
            <p><strong>Pipeline:</strong> {pipeline_name}</p>
            <p><strong>Failed At:</strong> {timestamp}</p>
            
            <h3>Error</h3>
            <div style="background-color: #ffeeee; padding: 15px; border-left: 4px solid #ff0000; margin: 15px 0;">
                <code>{error}</code>
            </div>
            
            {details_html}
            
            <h3>Action Required</h3>
            <p>Please check the logs at: <code>logs/pipeline.log</code></p>
            
            <hr style="margin-top: 30px;">
            <p style="color: #666; font-size: 12px;">
                PPP-Q ML Pipeline - Automated Notification
            </p>
        </body>
        </html>
        """
        
        self._send_email(subject, body)
        logger.error(f"Pipeline failure notification sent: {pipeline_name} - {error}")
    
    def notify_model_trained(self, model_metrics: Dict):
        """Send notification when model training completes"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        metrics_html = ''.join([
            f"<tr><td style='padding: 8px; border: 1px solid #ddd;'><strong>{k}</strong></td>"
            f"<td style='padding: 8px; border: 1px solid #ddd;'>{v}</td></tr>"
            for k, v in model_metrics.items()
        ])
        
        subject = "🎯 PPP-Q Model Training Complete"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #36a64f;">🎯 Model Training Complete</h2>
            <p><strong>Trained At:</strong> {timestamp}</p>
            
            <h3>Model Performance</h3>
            <table style="border-collapse: collapse; width: 100%; margin-top: 15px;">
                <tbody>
                    {metrics_html}
                </tbody>
            </table>
            
            <hr style="margin-top: 30px;">
            <p style="color: #666; font-size: 12px;">
                PPP-Q ML Pipeline - Automated Notification
            </p>
        </body>
        </html>
        """
        
        self._send_email(subject, body)
        logger.info("Model training notification sent")
    
    def notify_model_deployed(self, model_version: str, metrics: Dict):
        """Send notification when model is deployed"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        subject = f"🚀 PPP-Q Model v{model_version} Deployed"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #36a64f;">🚀 Model Deployed to Production</h2>
            <p><strong>Version:</strong> {model_version}</p>
            <p><strong>Deployed At:</strong> {timestamp}</p>
            
            <h3>Performance Metrics</h3>
            <ul>
                <li><strong>Macro F1:</strong> {metrics.get('macro_f1', 'N/A')}</li>
                <li><strong>Accuracy:</strong> {metrics.get('accuracy', 'N/A')}</li>
                <li><strong>Balanced Accuracy:</strong> {metrics.get('balanced_accuracy', 'N/A')}</li>
            </ul>
            
            <hr style="margin-top: 30px;">
            <p style="color: #666; font-size: 12px;">
                PPP-Q ML Pipeline - Automated Notification
            </p>
        </body>
        </html>
        """
        
        self._send_email(subject, body)
        logger.info(f"Model deployment notification sent: v{model_version}")
    
    def notify_data_ingestion(self, new_rows: int, total_rows: int):
        """Send notification about data ingestion"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        subject = "📊 PPP-Q Data Ingestion Complete"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #0099ff;">📊 Data Ingestion Complete</h2>
            <p><strong>Completed At:</strong> {timestamp}</p>
            
            <h3>Data Summary</h3>
            <ul>
                <li><strong>New Rows Added:</strong> {new_rows:,}</li>
                <li><strong>Total Rows:</strong> {total_rows:,}</li>
            </ul>
            
            <hr style="margin-top: 30px;">
            <p style="color: #666; font-size: 12px;">
                PPP-Q ML Pipeline - Automated Notification
            </p>
        </body>
        </html>
        """
        
        self._send_email(subject, body)
        logger.info(f"Data ingestion notification sent: {new_rows} new rows")
    
    def send_test_email(self):
        """Send a test email to verify configuration"""
        subject = "🧪 PPP-Q Pipeline Test Email"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #0099ff;">🧪 Test Email</h2>
            <p>This is a test email from the PPP-Q ML Pipeline.</p>
            <p><strong>Sent At:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>If you received this, your email notifications are configured correctly!</p>
            
            <hr style="margin-top: 30px;">
            <p style="color: #666; font-size: 12px;">
                PPP-Q ML Pipeline - Automated Notification
            </p>
        </body>
        </html>
        """
        
        return self._send_email(subject, body)


# Global notifier instance
notifier = NotificationManager()
