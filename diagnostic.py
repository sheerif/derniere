# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import cv2
import logging
import numpy as np
import time
import socket
from datetime import datetime
from openni import openni2
from openni import _openni2 as c_api
from logging.config import dictConfig
import os
import subprocess
import threading
import traceback
import queue

import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

import paramiko

# Définition des zones spécifiques dans l'image ou le flux vidéo
ZONE1 = {'nom': 'Zone Haut Gauche', 'coordonnees': (0, 0, 320, 240)}
ZONE2 = {'nom': 'Zone Bas Droite', 'coordonnees': (320, 240, 640, 480)}

# Fonction pour afficher les zones sur l'image
def afficher_zones(image, zoning_active):
    if not zoning_active:
        return image  # Ne pas afficher les zones si le zonage est désactivé
    for zone in [ZONE1, ZONE2]:
        x1, y1, x2, y2 = zone['coordonnees']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, zone['nom'], (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

# Fonction pour vérifier si des landmarks sont dans une zone
def landmarks_in_zone(landmarks, zone_coords, image_shape):
    x1, y1, x2, y2 = zone_coords
    img_height, img_width, _ = image_shape
    for landmark in landmarks.landmark:
        x = int(landmark.x * img_width)
        y = int(landmark.y * img_height)
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

# Fonction pour mesurer la qualité de l'image
def measure_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculer la variance du Laplacien
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

class CameraApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Cabine Connectée - Utilitaire de Diagnostic")

        # Configuration du logging
        self.setup_logging()
        self.logger = logging.getLogger('CameraApp')
        self.logger.debug("Application démarrée.")

        # Variables
        self.dev = None
        self.color_stream = None
        self.save_directory = "/home/Share/Enregistrements/"
        self.video_duration = 120  # Durée d'enregistrement en secondes (2 minutes)
        if self.video_duration <= 0:
            self.logger.error("La durée d'enregistrement (video_duration) doit être positive. Valeur actuelle: %s", self.video_duration)
            self.video_duration = 120  # Définir une valeur par défaut
        self.is_recording = False  # Indique si l'enregistrement est en cours
        self.flip_var = tk.BooleanVar(value=False)  # Variable pour la case à cocher "Retourner l'image"
        self.is_capturing_photo = False  # Indique si une photo est en cours de capture
        self.detect_person_var = tk.BooleanVar(value=False)  # Variable pour activer/désactiver la détection de personne
        self.is_live_feed_running = False  # Indique si le flux en direct est en cours
        self.model_complexity = 1  # Complexité fixe du modèle
        self.image_quality_scores = []  # Liste des scores de qualité d'image
        self.frame_count = 0  # Compteur de frames
        self.zoning_active_var = tk.BooleanVar(value=True)  # Le zonage est actif par défaut
        self.num_poste = self.get_num_poste()
        self.process_every_n_frames = 2  # Traiter une image sur N
        self.current_frame_index = 0
        self.resize_factor = 0.5  # Facteur de redimensionnement des images

        # Files d'attente pour la communication entre threads
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)

        # Événement pour arrêter le thread de traitement
        self.processing_thread_stop_event = threading.Event()

        # Création de l'interface utilisateur
        self.create_widgets()

        # Initialiser OpenNI2
        if not self.initialize_openni():
            messagebox.showerror("Erreur", "OpenNI2 n'a pas pu être initialisé.")
            return

        # Tenter d'initialiser la caméra au démarrage
        self.logger.info("Tentative d'initialisation de la caméra au démarrage.")
        if self.connect_cam():
            self.start_processing_thread()
            self.update_live_feed()
        else:
            self.logger.warning("Caméra non connectée au démarrage.")
            self.status_label.config(text="Caméra non connectée. Cliquez sur 'Vérifier la caméra'.")

    def setup_logging(self):
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {'simple': {'format': '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'}},
            'handlers': {
                'console': {'class': 'logging.StreamHandler', 'formatter': 'simple', 'level': logging.DEBUG},
                'file': {'class': 'logging.FileHandler', 'filename': 'diagnostic.log', 'formatter': 'simple', 'level': logging.DEBUG},
            },
            'root': {'handlers': ['console', 'file'], 'level': logging.DEBUG},
        }
        dictConfig(logging_config)

    def get_num_poste(self):
        hostname = socket.gethostname()
        num_poste = ''.join(filter(str.isdigit, hostname))
        return num_poste or '1'

    def create_widgets(self):
        main_frame = tk.Frame(self.master)
        main_frame.pack(padx=10, pady=10)

        nom_poste = f"POSTE : {socket.gethostname()}"
        label_nom_poste = tk.Label(main_frame, text=nom_poste, fg="red", font=("Courrier", 10))
        label_nom_poste.pack(pady=(0, 10))

        flip_checkbutton = tk.Checkbutton(main_frame, text="Retourner l'image", variable=self.flip_var)
        flip_checkbutton.pack()

        detect_person_checkbutton = tk.Checkbutton(main_frame, text="Activer la détection de personnes", variable=self.detect_person_var)
        detect_person_checkbutton.pack()

        zoning_checkbutton = tk.Checkbutton(main_frame, text="Activer le zonage", variable=self.zoning_active_var)
        zoning_checkbutton.pack()

        self.video_label = tk.Label(main_frame)
        self.video_label.pack()

        self.progress = ttk.Progressbar(main_frame, orient='horizontal', length=300, mode='determinate')
        self.progress.pack(pady=5)

        btn_check_cam = tk.Button(main_frame, text="Vérifier la caméra", command=self.test_cam)
        btn_check_cam.pack(fill='x')

        btn_record_video = tk.Button(main_frame, text="Enregistrer une vidéo avec détection de personnes", command=self.start_recording)
        btn_record_video.pack(fill='x', pady=5)

        btn_capture_photo = tk.Button(main_frame, text="Prendre une photo avec détection de personnes", command=self.capture_photo)
        btn_capture_photo.pack(fill='x', pady=5)

        btn_quit = tk.Button(main_frame, text="Quitter", command=self.on_closing, fg="red")
        btn_quit.pack(fill='x', pady=(5, 0))

        self.status_label = tk.Label(main_frame, text="", fg="blue")
        self.status_label.pack(pady=(5, 0))

    def initialize_openni(self):
        try:
            openni2.initialize()
            return True
        except Exception as e:
            self.logger.error("Erreur lors de l'initialisation d'OpenNI2: %s", e)
            return False

    def initialize_models(self):
        self.logger.debug("Initialisation des modèles MediaPipe.")
        self.mp_holistic = mp.solutions.holistic
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp_drawing_styles

        # Initialiser les modèles avec une complexité fixe
        self.holistic_model = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.pose_model = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.face_mesh_model = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def test_cam(self):
        self.logger.info("Début de la vérification de la caméra.")
        self.status_label.config(text="Vérification de la caméra...")
        self.master.update_idletasks()
        try:
            if self.is_live_feed_running:
                messagebox.showinfo("Caméra détectée", "La caméra est déjà connectée et le flux vidéo est actif.")
                self.logger.debug("Caméra déjà connectée et le flux vidéo est actif.")
                self.status_label.config(text="Caméra déjà connectée.")
                return

            self.logger.debug("Vérification de la présence de la caméra.")
            if self.check_cam() == 0:
                self.logger.warning("Caméra non détectée. Tentative de réinitialisation.")
                time.sleep(10)
                if self.check_cam() == 0:
                    messagebox.showerror("Erreur", "La caméra n'est toujours pas détectée après réinitialisation.")
                    self.logger.error("La caméra n'est pas détectée après réinitialisation.")
                    self.status_label.config(text="Caméra non détectée après réinitialisation.")
                    return
                else:
                    self.logger.info("Caméra détectée après réinitialisation.")

            self.logger.debug("Tentative de connexion de la caméra.")
            if self.connect_cam():
                messagebox.showinfo("Caméra détectée", "La caméra a été détectée et initialisée avec succès.")
                self.logger.debug("Caméra détectée et initialisée.")
                self.status_label.config(text="Caméra détectée et initialisée.")
                if not self.is_live_feed_running:
                    self.start_processing_thread()
                    self.update_live_feed()
            else:
                messagebox.showerror("Erreur", "Impossible de détecter ou d'initialiser la caméra.")
                self.logger.error("Erreur lors de la détection ou de l'initialisation de la caméra.")
                self.status_label.config(text="Erreur lors de la vérification de la caméra.")

        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue lors de la vérification de la caméra: {e}")
            self.logger.error("Erreur lors de la vérification de la caméra: %s", e)
            self.status_label.config(text="Erreur lors de la vérification de la caméra.")
        self.logger.info("Fin de la vérification de la caméra.")

    def start_processing_thread(self):
        self.processing_thread_stop_event.clear()
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()

        # Initialiser les modèles dans le thread de traitement
        self.initialize_models()

    def processing_loop(self):
        while not self.processing_thread_stop_event.is_set():
            try:
                # Récupérer une image de la file d'attente
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    continue

                img_rgb = frame['img_rgb']
                is_recording = frame['is_recording']
                flip = frame['flip']
                detect_person = frame['detect_person']
                zoning_active = frame['zoning_active']

                # Vérifier si l'image doit être retournée
                if flip:
                    img_rgb = cv2.flip(img_rgb, -1)

                # Mesurer la qualité de l'image
                image_quality = measure_image_quality(img_rgb)
                self.image_quality_scores.append(image_quality)
                self.frame_count += 1
                self.current_frame_index += 1

                if detect_person and self.current_frame_index % self.process_every_n_frames == 0:
                    # Traiter l'image
                    img_rgb = self.process_and_draw(img_rgb)
                else:
                    # Ne pas traiter l'image
                    if zoning_active:
                        img_rgb = afficher_zones(img_rgb, zoning_active)

                # Placer le résultat dans la file d'attente des résultats
                self.result_queue.put({'img_rgb': img_rgb})

                # Enregistrer la vidéo si l'enregistrement est actif
                if is_recording and self.video_out:
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    self.video_out.write(img_bgr)

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error("Erreur dans le thread de traitement: %s", e)
                self.logger.error(traceback.format_exc())

    def connect_cam(self):
        try:
            if self.dev is None:
                self.logger.debug("Ouverture du périphérique.")
                self.dev = openni2.Device.open_any()
            if self.color_stream is None:
                self.logger.debug("Création du flux couleur.")
                self.color_stream = self.dev.create_color_stream()
                self.logger.debug("Configuration du mode vidéo.")
                video_mode = c_api.OniVideoMode()
                video_mode.pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888
                video_mode.resolutionX = 640
                video_mode.resolutionY = 480
                video_mode.fps = 30
                self.color_stream.set_video_mode(video_mode)
                self.logger.debug("Démarrage du flux couleur.")
                self.color_stream.start()
            else:
                self.logger.debug("Le flux couleur est déjà démarré.")
            self.logger.debug("Caméra connectée et flux vidéo démarré.")
            return True
        except Exception as e:
            self.logger.error("Erreur lors de la connexion à la caméra: %s", e)
            self.logger.error(traceback.format_exc())
            self.disconnect_cam()
            messagebox.showerror("Erreur", f"Erreur lors de la connexion à la caméra : {e}")
            return False

    def disconnect_cam(self):
        self.is_recording = False
        self.processing_thread_stop_event.set()
        if self.color_stream:
            self.color_stream.stop()
            self.color_stream = None
        if self.dev:
            self.dev.close()
            self.dev = None
        self.logger.debug("Caméra déconnectée.")

    def update_live_feed(self):
        try:
            if not self.color_stream or self.is_capturing_photo:
                self.master.after(10, self.update_live_feed)
                return

            # Lire un nouveau cadre de couleur
            color_frame = self.color_stream.read_frame()
            color_frame_data = color_frame.get_buffer_as_uint8()
            color_img = np.frombuffer(color_frame_data, dtype=np.uint8).reshape((480, 640, 3))

            # Convertir l'image en RGB
            img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

            # Placer l'image dans la file d'attente pour le traitement
            frame_data = {
                'img_rgb': img_rgb,
                'is_recording': self.is_recording,
                'flip': self.flip_var.get(),
                'detect_person': self.detect_person_var.get(),
                'zoning_active': self.zoning_active_var.get()
            }
            try:
                self.frame_queue.put_nowait(frame_data)
            except queue.Full:
                pass  # Ignorer si la file d'attente est pleine

            # Récupérer le résultat du traitement
            try:
                result = self.result_queue.get_nowait()
                img_rgb = result['img_rgb']
            except queue.Empty:
                pass  # Pas de résultat disponible

            # Enregistrer la vidéo si l'enregistrement est actif
            if self.is_recording:
                elapsed_time = time.time() - self.record_start_time
                progress_percent = (elapsed_time / self.video_duration) * 100
                self.progress['value'] = progress_percent
                self.master.update_idletasks()
                if elapsed_time >= self.video_duration:
                    self.is_recording = False
                    if self.video_out:
                        self.video_out.release()
                        self.video_out = None
                    self.logger.info("Vidéo enregistrée avec succès : %s", self.video_filename)
                    self.status_label.config(text="Enregistrement terminé.")
                    messagebox.showinfo("Enregistrement terminé", f"Vidéo enregistrée : {self.video_filename}")
                    self.progress['value'] = 0

            # Afficher l'image dans l'interface Tkinter
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.imgtk = img_tk
            self.video_label.configure(image=img_tk)

        except Exception as e:
            self.logger.error("Erreur lors de la mise à jour du flux vidéo: %s", e)
            self.status_label.config(text="Erreur lors de la mise à jour du flux vidéo.")

        # Planifier la prochaine mise à jour
        self.master.after(10, self.update_live_feed)

    def process_and_draw(self, img_rgb):
        try:
            # Redimensionner l'image
            img_small = cv2.resize(img_rgb, (0, 0), fx=self.resize_factor, fy=self.resize_factor)

            # Traiter l'image avec les modèles MediaPipe
            holistic_results = self.holistic_model.process(img_small)
            pose_results = self.pose_model.process(img_small)
            face_mesh_results = self.face_mesh_model.process(img_small)

            # Dessiner les landmarks sur l'image originale en ajustant les coordonnées
            self.draw_landmarks(img_rgb, holistic_results, pose_results, face_mesh_results, scale=(1 / self.resize_factor))
            return img_rgb
        except Exception as e:
            self.logger.error("Erreur lors du traitement de l'image: %s", e)
            self.logger.error(traceback.format_exc())
            return img_rgb

    def draw_landmarks(self, img_rgb, holistic_results, pose_results, face_mesh_results, scale=1):
        person_detected = False

        # Fonction pour ajuster les coordonnées des landmarks
        def scale_landmarks(landmarks, scale):
            for landmark in landmarks.landmark:
                landmark.x *= scale
                landmark.y *= scale
                landmark.z *= scale
            return landmarks

        # Dessiner les résultats du modèle Holistic
        if holistic_results and holistic_results.pose_landmarks:
            scaled_landmarks = scale_landmarks(holistic_results.pose_landmarks, scale)
            self.mp_drawing.draw_landmarks(
                img_rgb,
                scaled_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            person_detected = True

        # Dessiner les résultats du modèle Pose
        if pose_results and pose_results.pose_landmarks:
            scaled_landmarks = scale_landmarks(pose_results.pose_landmarks, scale)
            self.mp_drawing.draw_landmarks(
                img_rgb,
                scaled_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            person_detected = True

        # Dessiner les résultats du modèle Face Mesh
        if face_mesh_results and face_mesh_results.multi_face_landmarks:
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                scaled_landmarks = scale_landmarks(face_landmarks, scale)
                self.mp_drawing.draw_landmarks(
                    image=img_rgb,
                    landmark_list=scaled_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
            person_detected = True

        # Vérifier le zonage
        if self.zoning_active_var.get():
            zones_detectees = []
            landmarks_to_check = None
            if holistic_results and holistic_results.pose_landmarks:
                landmarks_to_check = holistic_results.pose_landmarks
            elif pose_results and pose_results.pose_landmarks:
                landmarks_to_check = pose_results.pose_landmarks

            if landmarks_to_check:
                for zone in [ZONE1, ZONE2]:
                    if landmarks_in_zone(landmarks_to_check, zone['coordonnees'], img_rgb.shape):
                        zones_detectees.append(zone['nom'])

            # Dessiner les zones sur l'image
            img_rgb = afficher_zones(img_rgb, self.zoning_active_var.get())

            # Afficher un message sur l'image
            if zones_detectees:
                message = f"Personne détectée dans : {', '.join(zones_detectees)}"
                cv2.putText(img_rgb, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                self.logger.info(message)
            else:
                cv2.putText(img_rgb, "Aucune personne détectée dans les zones", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.logger.warning("Aucune personne détectée dans les zones.")
        else:
            if person_detected:
                cv2.putText(img_rgb, "Personne détectée dans le cadre", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                self.logger.info("Personne détectée dans le cadre.")
            else:
                cv2.putText(img_rgb, "Aucune personne détectée", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.logger.warning("Aucune personne détectée dans le cadre.")

    def start_recording(self):
        if self.is_recording:
            messagebox.showwarning("Enregistrement en cours", "Un enregistrement est déjà en cours.")
            return
        self.logger.info("Début de l'enregistrement vidéo.")
        self.status_label.config(text="Démarrage de l'enregistrement vidéo...")
        self.master.update_idletasks()
        if not self.dev or not self.color_stream:
            messagebox.showerror("Erreur", "La caméra n'est pas initialisée.")
            self.logger.error("La caméra n'est pas initialisée.")
            return
        self.is_recording = True
        now = datetime.now()
        date_str = now.strftime("%d_%m_%Y_%H_%M_%S")
        self.video_filename = os.path.join(self.save_directory, f"video_{date_str}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_out = cv2.VideoWriter(self.video_filename, fourcc, 30.0, (640, 480))
        self.record_start_time = time.time()
        self.progress['value'] = 0

    def capture_photo(self):
        self.logger.info("Début de la capture de la photo.")
        self.is_capturing_photo = True
        self.status_label.config(text="Capture de la photo...")
        self.master.update_idletasks()
        try:
            if not self.dev or not self.color_stream:
                messagebox.showerror("Erreur", "La caméra n'est pas initialisée.")
                self.logger.error("La caméra n'est pas initialisée.")
                return

            color_frame = self.color_stream.read_frame()
            color_frame_data = color_frame.get_buffer_as_uint8()
            color_img = np.frombuffer(color_frame_data, dtype=np.uint8).reshape((480, 640, 3))
            img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

            # Traiter l'image dans le thread de traitement
            frame_data = {
                'img_rgb': img_rgb,
                'is_recording': False,
                'flip': self.flip_var.get(),
                'detect_person': self.detect_person_var.get(),
                'zoning_active': self.zoning_active_var.get()
            }
            self.frame_queue.put(frame_data)
            time.sleep(0.1)  # Attendre un court instant pour que le traitement se fasse
            result = self.result_queue.get(timeout=1)
            img_rgb = result['img_rgb']

            now = datetime.now()
            date_str = now.strftime("%d_%m_%Y_%H_%M_%S")
            image_filename = os.path.join(self.save_directory, f"photo_{date_str}.png")
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_filename, img_bgr)
            self.logger.info("Photo enregistrée avec succès : %s", image_filename)
            messagebox.showinfo("Photo prise", f"Photo enregistrée : {image_filename}")
            self.status_label.config(text="Photo capturée.")

            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.imgtk = img_tk
            self.video_label.configure(image=img_tk)

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la capture de la photo : {e}")
            self.logger.error("Erreur lors de la capture de la photo: %s", e)
            self.status_label.config(text="Erreur lors de la capture de la photo.")
        finally:
            self.is_capturing_photo = False
            self.logger.info("Fin de la capture de la photo.")

    def run(self):
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.master.mainloop()

    def on_closing(self):
        if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter l'application ?"):
            self.disconnect_cam()
            self.close_models()
            self.master.destroy()
            self.logger.info("Application fermée par l'utilisateur.")
            self.status_label.config(text="Application fermée.")
            self.logger.info("Fin de l'application.")

    def close_models(self):
        try:
            if self.holistic_model:
                self.holistic_model.close()
                self.holistic_model = None
            if self.pose_model:
                self.pose_model.close()
                self.pose_model = None
            if self.face_mesh_model:
                self.face_mesh_model.close()
                self.face_mesh_model = None
        except Exception as e:
            self.logger.error("Erreur lors de la fermeture des modèles: %s", e)

    def cmd_terminal_local(self, cmd):
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Erreur lors de l'exécution de la commande : {cmd}\n{e}")

    def init_anyusb(self):
        client = None
        try:
            ip_anyusb = 60 if (1 <= int(self.num_poste) <= 8) else 61
            portusb = int(self.num_poste) if (1 <= int(self.num_poste) <= 8) else int(self.num_poste) - 8

            hostname = f'10.10.10.{ip_anyusb}'
            username = 'admin'
            password = 'Masternaute2023*'  # **Remplacez ceci par votre mot de passe SSH réel**

            self.logger.info(f"init_anyusb - Connexion SSH à {hostname} sur le port USB {portusb}...")
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname, username=username, password=password)

            commande = f'sudo system anywhereusb powercycle port{portusb}'
            self.logger.info(f"init_anyusb - Exécution de la commande: {commande}")
            stdin, stdout, stderr = client.exec_command(commande)
            stdout.channel.recv_exit_status()
            self.logger.info("init_anyusb - Reboot caméra en cours...")
            client.close()
            time.sleep(10)
            self.logger.info("init_anyusb - Redémarrage de la caméra FAIT!")

        except paramiko.AuthenticationException:
            self.logger.error("init_anyusb - Échec de l'authentification SSH. Vérifiez votre nom d'utilisateur et votre mot de passe.")
        except paramiko.SSHException as sshException:
            self.logger.error(f"init_anyusb - Problème de connexion SSH : {sshException}")
        except Exception as e:
            self.logger.exception("init_anyusb - Erreur lors de l'initialisation d'AnywhereUSB")
            if client:
                client.close()

    def check_cam(self):
        try:
            command1 = 'lsusb | grep "Orbbec" > list_cam.txt'
            command2 = 'cp list_cam.txt /home/Share/list_cam.txt'
            self.logger.info("Exécution des commandes de vérification de la caméra...")
            self.cmd_terminal_local(command1)
            self.cmd_terminal_local(command2)

            list_cam = "/home/Share/list_cam.txt"
            with open(list_cam, "r") as f:
                content = f.read()
            cam_or_not = len(content)
            if cam_or_not == 0:
                self.logger.warning("Pas de caméra détectée")
                self.init_anyusb()
            else:
                self.logger.info("Caméra détectée")

            return cam_or_not
        except Exception as e:
            self.logger.exception("Erreur lors de la vérification de la caméra")
            return 0

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = CameraApp(root)
        app.run()
    except Exception as e:
        logging.error("Erreur fatale: %s", e)
        logging.error(traceback.format_exc())
    finally:
        openni2.unload()
        logging.info("Programme terminé.")
