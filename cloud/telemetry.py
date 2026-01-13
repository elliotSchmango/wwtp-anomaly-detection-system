import json
import logging
import sqlite3
import threading
import time
import os
from pathlib import Path
from azure.iot.device import IoTHubDeviceClient, Message
from config.settings import settings

class DurableQueue:
    """Simple FIFO queue backed by SQLite for offline buffering"""
    def __init__(self, db_path):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                payload TEXT
            )
        ''')
        self.conn.commit()
        self.lock = threading.Lock()

    def enqueue(self, payload: dict):
        with self.lock:
            self.conn.execute('INSERT INTO queue (payload) VALUES (?)', (json.dumps(payload),))
            self.conn.commit()

    def peek(self):
        with self.lock:
            cursor = self.conn.execute('SELECT id, payload FROM queue ORDER BY id ASC LIMIT 1')
            row = cursor.fetchone()
            return row if row else None

    def pop(self, item_id):
        with self.lock:
            self.conn.execute('DELETE FROM queue WHERE id = ?', (item_id,))
            self.conn.commit()
            
    def close(self):
        self.conn.close()

class TelemetrySender:
    def __init__(self):
        self.logger = logging.getLogger("Eagle.Azure")
        self.client = None
        self.enabled = bool(settings.IOTHUB_CONN_STRING)
        self.queue = None
        self.flusher_thread = None
        self.stop_event = threading.Event()

        if self.enabled:
            queue_path = os.getenv("QUEUE_DB", str(settings.DATA_DIR / "queue.db"))
            self.queue = DurableQueue(queue_path)

            try:
                self.client = IoTHubDeviceClient.create_from_connection_string(
                    settings.IOTHUB_CONN_STRING
                )
                self.client.connect()
                self.logger.info("Connected to Azure IoT Hub")
                
                # 3. Start Background Flusher
                self.flusher_thread = threading.Thread(target=self._flush_loop, daemon=True)
                self.flusher_thread.start()
                
            except Exception as e:
                self.logger.error(f"Failed to connect to Azure: {e}")
                self.enabled = False

    #Public API: attempt to send immediately, queue on failure
    def send_payload(self, payload: dict):
        if not self.enabled:
            return
        
        if not self._send_to_hub(payload):
            self.logger.warning("Connection unstable. Buffering alert to disk.")
            self.queue.enqueue(payload)

    def _send_to_hub(self, payload) -> bool:
        if not self.client:
            return False
            
        try:
            msg = Message(json.dumps(payload))
            msg.content_type = "application/json"
            msg.content_encoding = "utf-8"
            msg.custom_properties["is_anomaly"] = "true" if payload.get("anomaly_flag") else "false"
            
            self.client.send_message(msg)
            return True
        except Exception as e:
            self.logger.error(f"Transmission failed: {e}")
            return False

    def _flush_loop(self):
        self.logger.info("Background flusher started")
        while not self.stop_event.is_set():
            # Check queue
            item = self.queue.peek()
            if item:
                item_id, payload_json = item
                try:
                    payload = json.loads(payload_json)
                    if self._send_to_hub(payload):
                        self.queue.pop(item_id)
                        self.logger.info(f"🔄 Flushed queued item {item_id}")
                    else:
                        time.sleep(5)
                except Exception as e:
                    self.logger.error(f"Queue Error: {e}")
                    self.queue.pop(item_id)
            else:
                #queue is empty
                time.sleep(2)

    def disconnect(self):
        self.stop_event.set()
        if self.flusher_thread:
            self.flusher_thread.join(timeout=2)
        if self.client:
            self.client.shutdown()
        if self.queue:
            self.queue.close()
