#!/usr/bin/python
# -*- coding: utf-8 -*-

# =============================================
# Importations
# =============================================
import os
import sys
import subprocess
import threading
import time
import socket
import logging
import signal
from datetime import datetime
from math import degrees, acos
from typing import List, Optional, Dict
from enum import Enum

import cv2
import numpy as np
import mediapipe as mp
import paramiko
from termcolor import colored
from openni import openni2
import platform
from collections import deque, namedtuple
from logging.handlers import RotatingFileHandler  # Import pour la rotation des logs

# =============================================
# Informations de Version
# =============================================
__version__ = "1.0.8"  # Mise à jour de la version

# =============================================
# Configuration du Logging
# =============================================

# Nom de fichier de log fixe
log_filename = "CameraApplication.log"

# Configuration du logging avec rotation des fichiers (optionnel mais recommandé)
handler = RotatingFileHandler(log_filename, maxBytes=5*1024*1024, backupCount=5)  # 5 Mo par fichier, 5 sauvegardes
logging.basicConfig(
    level=logging.DEBUG,  # Niveau de log défini à DEBUG pour le maximum d'informations
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',  # Format des logs
    handlers=[
        handler,
        logging.StreamHandler(sys.stdout)  # Affichage des logs sur la console
    ]
)

logging.info("Configuration du logging réussie.")
logging.info(f"Début de l'application CameraApplication, Version {__version__}")

# =============================================
# Définition des namedtuples
# =============================================

# Exemple de namedtuple pour stocker les points clés
KeyPoint = namedtuple('KeyPoint', ['x', 'y'])

# =============================================
# Définition des KeyPoints avec Enum
# =============================================

class KeyPoints(Enum):
    """Énumération des points clés utilisés pour l'analyse de la posture."""
    CHEST_FRONT = 0
    CHEST_ROTATION = 1
    CHEST_SIDE = 2
    ELBOW_LEFT = 3
    ELBOW_RIGHT = 4
    SHOULDER_LEFT_ROTATION = 5
    SHOULDER_LEFT_RAISING = 6
    SHOULDER_RIGHT_ROTATION = 7
    SHOULDER_RIGHT_RAISING = 8
    NECK_FLEXION = 9
    WRIST_LEFT = 10
    WRIST_RIGHT = 11
    HIP_LEFT = 12
    HIP_RIGHT = 13
    KNEE_LEFT = 14
    KNEE_RIGHT = 15
    ANKLE_LEFT = 16
    ANKLE_RIGHT = 17

# =============================================
# Manager de KeyPoints avec thread-safety
# =============================================

class KeyPointManager:
    """Classe pour gérer les KeyPoints de manière thread-safe."""

    def __init__(self, lock: threading.Lock):
        self.lock = lock
        self.keypoints = deque(maxlen=1000)  # Limitation de la taille pour éviter une croissance infinie

    def add_keypoint(self, x: float, y: float) -> None:
        """Ajoute un KeyPoint de manière thread-safe."""
        with self.lock:
            kp = KeyPoint(x, y)
            self.keypoints.append(kp)
            logging.debug(f"KeyPoint ajouté: {kp}")

    def get_keypoints(self) -> List[KeyPoint]:
        """Retourne une copie des KeyPoints."""
        with self.lock:
            return list(self.keypoints)

# =============================================
# Classe Principale
# =============================================

class CameraApplication:
    """Classe principale pour gérer l'application de la caméra."""

    def __init__(self):
        # Ignorer le signal SIGPIPE pour éviter que le processus ne se termine brusquement
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)

        # ***** Variables/Paramètres *****
        self.plage_ip = "10.10.10."
        self.ip_concentrateur = 70
        self.full_ip_concentrateur = self.plage_ip + str(self.ip_concentrateur)
        
        self.repertoire_sauvegarde = "/home/Share/Enregistrements/"
        self.nom_poste = socket.gethostname()
        self.num_poste = ''.join(filter(str.isdigit, self.nom_poste))
        self.nom_du_poste = f"Caméra n° : {self.num_poste}"
        self.app_is_on = "no"
        self.recording = "no"
        self.pres_cam = "no"
        self.result_analyse = "_0_1_2_3_4_5_6_7_8_9"

        # Variables de synchronisation
        self.stop_event = threading.Event()
        self.periodic_thread = None

        # Paramètres de reconnexion
        self.max_retries = 5
        self.retry_delay = 5  # Secondes

        # ***** Attributs pour la Gestion des Scores *****
        self.action_history = deque(maxlen=1000)  # Historique des actions avec timestamps
        self.repetitivite_window = 60  # Fenêtre de temps en secondes pour le calcul

        self.posture_history = deque(maxlen=1000)  # Historique des postures avec timestamps
        self.posture_window = 60  # Fenêtre de temps en secondes pour le calcul

        self.last_activity_time = None
        self.recuperation_threshold = 30  # Temps de récupération en secondes

        self.hand_positions_history = deque(maxlen=1000)  # Historique des positions des poignets avec timestamps
        self.prehension_window = 60  # Fenêtre de temps en secondes pour le calcul

        # ***** Attributs pour la Gestion Dynamique de la Complexité du Modèle *****
        self.false_positives = 0
        self.false_negatives = 0
        self.performance_history = deque(maxlen=1000)  # Historique des performances avec timestamps
        self.performance_window = 60  # Fenêtre de temps en secondes pour le calcul des performances
        self.adjustment_threshold = 5  # Nombre d'erreurs pour ajuster la complexité

        self.consecutive_no_detection_frames = 0  # Compteur de frames sans détection
        self.desired_model_complexity = 1  # Complexité souhaitée du modèle, initialisée à 1

        # ***** Locks pour la Sécurité des Threads *****
        self.camera_lock = threading.Lock()
        self.mediapipe_lock = threading.Lock()
        self.keypoint_lock = threading.Lock()

        # ***** Initialisation de Mediapipe *****
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.3,  # Ajusté pour une détection plus tolérante
            min_tracking_confidence=0.3,
            model_complexity=1  # Initialisé à 1 pour commencer
        )
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.3,  # Ajusté pour une détection plus tolérante
            min_tracking_confidence=0.3
        )
        self.current_model_complexity = 1  # Attribut pour suivre la complexité actuelle du modèle

        # Compteur pour le Message Data Version (MDV)
        self.mdv = 0

        logging.info("*************** NE PAS FERMER ***************")
        logging.info("********** INITIALISATION **********")

        # ***** Initialisation du Manager de KeyPoints *****
        self.keypoint_manager = KeyPointManager(self.keypoint_lock)

    # =============================================
    # Gestion des Signaux
    # =============================================

    def signal_handler(self, sig, frame) -> None:
        """Gestionnaire de signaux pour un arrêt gracieux."""
        logging.info('Signal reçu. Fermeture en douceur...')
        self.sortie_programme()

    # =============================================
    # Fonctions Réseau et Périphériques
    # =============================================

    def ping_host(self, host: str) -> bool:
        """Vérifie la connectivité avec un hôte en envoyant un ping."""
        try:
            param = '-n' if platform.system().lower() == 'windows' else '-c'
            command = ['ping', param, '1', host]
            logging.info(f"Envoi d'un ping à {host}...")
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                logging.info(f"Ping réussi vers {host}.")
                return True
            else:
                logging.warning(f"Ping échoué vers {host}.")
                return False
        except Exception as e:
            logging.exception(f"Erreur lors du ping de l'hôte {host}")
            return False

    def init_anyusb(self) -> None:
        """Initialise et redémarre le périphérique AnywhereUSB."""
        client = None
        try:
            ip_anyusb = 60 if (1 <= int(self.num_poste) <= 8) else 61
            portusb = int(self.num_poste) if (1 <= int(self.num_poste) <= 8) else int(self.num_poste) - 8
            
            hostname = f'10.10.10.{ip_anyusb}'
            username = 'admin'  # Nom d'utilisateur SSH
            password = 'Masternaute2023*'  # **Remplacez ceci par votre mot de passe SSH réel**
            
            logging.info(f"init_anyusb - Connexion SSH à {hostname} sur le port USB {portusb}...")
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname, username=username, password=password)
            
            commande = f'sudo system anywhereusb powercycle port{portusb}'
            logging.info(f"init_anyusb - Exécution de la commande: {commande}")
            stdin, stdout, stderr = client.exec_command(commande)
            stdout.channel.recv_exit_status()  # Attendre la fin de la commande
            logging.info("init_anyusb - Reboot caméra en cours...")
            client.close()
            time.sleep(10)
            logging.info("init_anyusb - Redémarrage de la caméra FAIT!")
            
        except paramiko.AuthenticationException:
            logging.error("init_anyusb - Échec de l'authentification SSH. Vérifiez votre nom d'utilisateur et votre mot de passe.")
        except paramiko.SSHException as sshException:
            logging.error(f"init_anyusb - Problème de connexion SSH : {sshException}")
        except Exception as e:
            logging.exception("init_anyusb - Erreur lors de l'initialisation d'AnywhereUSB")
            if client:
                try:
                    client.close()
                except:
                    pass

    def check_cam(self) -> int:
        """Vérifie si la caméra est détectée sur le système."""
        try:
            command1 = 'lsusb | grep "Orbbec" > list_cam.txt'
            command2 = 'cp list_cam.txt /home/Share/list_cam.txt'
            logging.info("Exécution des commandes de vérification de la caméra...")
            self.cmd_terminal_local(command1)
            self.cmd_terminal_local(command2)

            list_cam = "/home/Share/list_cam.txt"
            with open(list_cam, "r") as f:
                content = f.read()
            cam_or_not = len(content)
            if cam_or_not == 0:
                logging.warning(colored("Pas de caméra détectée", "red"))
                self.init_anyusb()
            else:
                logging.info(colored("Caméra détectée", "green"))

            return cam_or_not
        except Exception as e:
            logging.exception("Erreur lors de la vérification de la caméra")
            return 0

    # =============================================
    # Fonctions Utilitaires
    # =============================================

    def mdv_app(self) -> None:
        """Incrémente le compteur MDV."""
        try:
            self.mdv = (self.mdv + 1) % 60
            logging.debug(f"MDV mis à jour: {self.mdv}")
        except Exception as e:
            logging.exception("Erreur lors de la mise à jour du MDV")

    def cmd_terminal_local(self, command: str) -> None:
        """Exécute une commande dans le terminal local."""
        try:
            logging.info(f"Exécution de la commande: {command}")
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Échec de la commande: {command}")
            logging.exception(e)
        except Exception as e:
            logging.exception(f"Erreur lors de l'exécution de la commande: {command}")

    # =============================================
    # Gestion de la Caméra
    # =============================================

    def record_image(self) -> str:
        """Enregistre une image à partir du flux de la caméra."""
        try:
            with self.camera_lock:
                logging.info("Lecture d'une frame depuis le flux couleur...")
                color_frame = self.color_stream.read_frame()
                color_frame_data = color_frame.get_buffer_as_uint8()
                color_img = np.frombuffer(color_frame_data, dtype=np.uint8)
                
                video_mode = self.color_stream.get_video_mode()
                width = video_mode.resolutionX
                height = video_mode.resolutionY
                expected_size = width * height * 3  # 3 pour RGB

                logging.debug(f"Taille des données de frame: {len(color_frame_data)}")
                if len(color_frame_data) < expected_size:
                    logging.error("Données de frame insuffisantes pour le reshaping.")
                    return ""
                
                color_img = color_img.reshape((height, width, 3))
                logging.debug(f"Forme de color_img après reshaping: {color_img.shape}")
                
                # Validation de la forme de l'image
                if color_img.shape != (height, width, 3):
                    logging.error(f"Forme de l'image incorrecte après reshaping: {color_img.shape}")
                    return ""
                if not np.all((color_img >= 0) & (color_img <= 255)):
                    logging.error("Valeurs de l'image hors limites après conversion.")
                    return ""
                
                color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                
                now = datetime.now()
                date = now.strftime("%d_%m_%Y_%H_%M_%S_%f")
                filename = os.path.join(self.repertoire_sauvegarde, f"img_{date}.jpg")
                
                cv2.imwrite(filename, color_img)
                logging.debug(f"Image enregistrée: {filename}")
                with open("Last_img.txt", "w") as fichier:
                    fichier.write(filename)
                return filename
        except Exception as e:
            logging.exception("Erreur lors de l'enregistrement de l'image")
            return ""

    def sortie_programme(self) -> None:
        """Ferme proprement toutes les ressources et quitte le programme."""
        try:
            # Signaler aux threads de s'arrêter
            self.stop_event.set()
            
            if self.periodic_thread and self.periodic_thread.is_alive():
                self.periodic_thread.join(timeout=5)
                logging.info("Thread périodique arrêté.")
            
            if hasattr(self, 'color_stream') and self.color_stream and self.color_stream.is_valid:
                logging.info("Arrêt du flux couleur...")
                self.color_stream.stop()
                logging.info("Flux couleur arrêté.")
            if hasattr(self, 'dev') and self.dev:
                logging.info("Déchargement de OpenNI...")
                openni2.unload()
                logging.info("OpenNI déchargé.")
            if hasattr(self, 'holistic') and self.holistic:
                logging.info("Fermeture de Mediapipe Holistic...")
                with self.mediapipe_lock:
                    self.holistic.close()
                logging.info("Mediapipe Holistic fermé.")
            if hasattr(self, 'pose') and self.pose:
                logging.info("Fermeture de Mediapipe Pose...")
                with self.mediapipe_lock:
                    self.pose.close()
                logging.info("Mediapipe Pose fermé.")
            cv2.destroyAllWindows()
            logging.info("Fenêtres OpenCV fermées.")
        except Exception as e:
            logging.exception("Erreur lors de la fermeture du programme")
        finally:
            sys.exit(0)

    # =============================================
    # Traitement des Images et Analyse
    # =============================================

    def calculate_image_quality(self, image: np.ndarray) -> float:
        """
        Calculer la qualité de l'image en fonction de la luminosité et de la netteté.
        Utilisé pour ajuster la complexité du modèle MediaPipe.
        """
        if image is None or not isinstance(image, np.ndarray):
            logging.warning("Image invalide pour le calcul de la qualité")
            return 0.0

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray) / 255.0
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = variance / (variance + 1e-6)  # Éviter la division par zéro
            quality = min(1.0, max(0.0, 0.5 * brightness + 0.5 * (sharpness / (sharpness + 1))))
            logging.debug(f"Qualité de l'image: {quality}")
            return quality
        except Exception as e:
            logging.exception("Erreur lors du calcul de la qualité de l'image")
            return 0.0

    def calculate_angle(self, a: List[float], b: List[float], c: List[float]) -> int:
        """
        Calculer l'angle en degrés formé par trois points a, b, c.
        L'angle est au point b entre les segments ba et bc.
        """
        try:
            ba = np.array(a) - np.array(b)
            bc = np.array(c) - np.array(b)
            norm_ba = np.linalg.norm(ba)
            norm_bc = np.linalg.norm(bc)
            if norm_ba == 0 or norm_bc == 0:
                logging.warning("Vecteur de norme zéro détecté lors du calcul de l'angle")
                return 0
            dot_product = np.dot(ba, bc)
            cos_angle = dot_product / (norm_ba * norm_bc)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = int(degrees(acos(cos_angle)))
            logging.debug(f"Angle calculé: {angle} degrés")
            return angle
        except Exception as e:
            logging.exception("Erreur lors du calcul de l'angle")
            return 0

    def preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Prétraiter l'image pour améliorer la détection des points clés.
        """
        if image is None or not isinstance(image, np.ndarray):
            logging.warning("Image invalide pour le prétraitement")
            return None

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
            processed_image = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
            logging.debug("Image prétraitée avec succès")
            return processed_image
        except Exception as e:
            logging.exception("Erreur lors du prétraitement de l'image")
            return None

    def classify_angle(self, angle: int, thresholds: Dict[str, any]) -> int:
        """
        Classer un angle donné dans une zone de risque selon les seuils fournis.
        """
        try:
            if thresholds['green'][0] <= angle <= thresholds['green'][1]:
                return 1  # Zone verte (faible risque)
            elif any(lower <= angle <= upper for (lower, upper) in thresholds['orange']):
                return 2  # Zone orange (risque modéré)
            elif angle >= thresholds['red'][0]:
                return 3  # Zone rouge (risque élevé)
            else:
                return 0  # Valeur par défaut si aucun seuil n'est satisfait
        except Exception as e:
            logging.exception("Erreur lors de la classification de l'angle")
            return 0

    def extract_keypoints(self, landmarks: List[mp.solutions.holistic.PoseLandmark]) -> Dict[KeyPoints, Optional[List[float]]]:
        """
        Extraire les points clés nécessaires des landmarks détectés par Mediapipe.
        """
        try:
            def get_landmark_value(landmark):
                if landmark:
                    return [landmark.x, landmark.y]
                else:
                    return None

            keypoints = {
                KeyPoints.SHOULDER_LEFT_ROTATION: None,
                KeyPoints.ELBOW_LEFT: None,
                KeyPoints.WRIST_LEFT: None,
                KeyPoints.SHOULDER_RIGHT_ROTATION: None,
                KeyPoints.ELBOW_RIGHT: None,
                KeyPoints.WRIST_RIGHT: None,
                KeyPoints.NECK_FLEXION: None,
                KeyPoints.HIP_LEFT: None,
                KeyPoints.HIP_RIGHT: None,
                KeyPoints.KNEE_LEFT: None,
                KeyPoints.KNEE_RIGHT: None,
                KeyPoints.ANKLE_LEFT: None,
                KeyPoints.ANKLE_RIGHT: None,
            }

            if landmarks:
                # Extraction des points clés
                keypoints[KeyPoints.SHOULDER_LEFT_ROTATION] = get_landmark_value(landmarks[self.mp_holistic.PoseLandmark.LEFT_SHOULDER])
                keypoints[KeyPoints.ELBOW_LEFT] = get_landmark_value(landmarks[self.mp_holistic.PoseLandmark.LEFT_ELBOW])
                keypoints[KeyPoints.WRIST_LEFT] = get_landmark_value(landmarks[self.mp_holistic.PoseLandmark.LEFT_WRIST])
                keypoints[KeyPoints.SHOULDER_RIGHT_ROTATION] = get_landmark_value(landmarks[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER])
                keypoints[KeyPoints.ELBOW_RIGHT] = get_landmark_value(landmarks[self.mp_holistic.PoseLandmark.RIGHT_ELBOW])
                keypoints[KeyPoints.WRIST_RIGHT] = get_landmark_value(landmarks[self.mp_holistic.PoseLandmark.RIGHT_WRIST])
                keypoints[KeyPoints.HIP_LEFT] = get_landmark_value(landmarks[self.mp_holistic.PoseLandmark.LEFT_HIP])
                keypoints[KeyPoints.HIP_RIGHT] = get_landmark_value(landmarks[self.mp_holistic.PoseLandmark.RIGHT_HIP])
                keypoints[KeyPoints.KNEE_LEFT] = get_landmark_value(landmarks[self.mp_holistic.PoseLandmark.LEFT_KNEE])
                keypoints[KeyPoints.KNEE_RIGHT] = get_landmark_value(landmarks[self.mp_holistic.PoseLandmark.RIGHT_KNEE])
                keypoints[KeyPoints.ANKLE_LEFT] = get_landmark_value(landmarks[self.mp_holistic.PoseLandmark.LEFT_ANKLE])
                keypoints[KeyPoints.ANKLE_RIGHT] = get_landmark_value(landmarks[self.mp_holistic.PoseLandmark.RIGHT_ANKLE])

                # Calcul du point du cou (NECK_FLEXION)
                left_shoulder = keypoints[KeyPoints.SHOULDER_LEFT_ROTATION]
                right_shoulder = keypoints[KeyPoints.SHOULDER_RIGHT_ROTATION]

                if left_shoulder and right_shoulder:
                    neck = [
                        (left_shoulder[0] + right_shoulder[0]) / 2,
                        (left_shoulder[1] + right_shoulder[1]) / 2
                    ]
                    keypoints[KeyPoints.NECK_FLEXION] = neck
                else:
                    keypoints[KeyPoints.NECK_FLEXION] = None

            return keypoints
        except Exception as e:
            logging.exception("Erreur lors de l'extraction des keypoints")
            return {}

    def detect_actions_techniques_in_image(self, image: np.ndarray) -> List[tuple]:
        """
        Détecter des "actions techniques" dans l'image.
        """
        try:
            logging.info("Début de la détection des actions techniques dans l'image.")
            
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            image_normalized = image / 255.0
            actions_mask = np.zeros_like(image_normalized, dtype=bool)
            
            if image_normalized.shape[0] < 3 or image_normalized.shape[1] < 3:
                logging.error("L'image est trop petite pour effectuer le slicing requis.")
                return []
            
            actions_mask[1:-1, 1:-1] = (
                (image_normalized[1:-1, 1:-1] > image_normalized[:-2, 1:-1]) &
                (image_normalized[1:-1, 1:-1] > image_normalized[2:, 1:-1]) &
                (image_normalized[1:-1, 1:-1] > image_normalized[1:-1, :-2]) &
                (image_normalized[1:-1, 1:-1] > image_normalized[1:-1, 2:])
            )
            
            actions = np.argwhere(actions_mask)
            actions = [(int(i), int(j)) for i, j in actions]
            logging.debug(f"Nombre d'actions techniques détectées: {len(actions)}")
            return actions
        except Exception as e:
            logging.exception("Erreur lors de la détection des actions techniques")
            return []

    def estimator(self, image_path: str) -> str:
        """
        Fonction principale pour analyser la posture dans une image donnée.
        Retourne une chaîne de caractères formatée avec les résultats de l'analyse.
        """
        # Initialisation des valeurs par défaut
        fields = [0] * 10  # Liste de 10 éléments initialisés à 0

        # Définition des seuils pour la classification
        thresholds = {
            'flexion_cou': {
                'green': (0, 10),
                'orange': [(11, 20), (21, 30)],
                'red': (31, 90)
            },
            'flexion_epaule_gauche': {
                'green': (0, 20),
                'orange': [(21, 45)],
                'red': (46, 180)
            },
            'flexion_epaule_droite': {
                'green': (0, 20),
                'orange': [(21, 45)],
                'red': (46, 180)
            },
            'flexion_tronc': {
                'green': (0, 20),
                'orange': [(21, 45)],
                'red': (46, 90)
            },
            'flexion_genou': {
                'green': (0, 30),
                'orange': [(31, 60)],
                'red': (61, 180)
            },
            'abduction_epaule_gauche': {
                'green': (0, 20),
                'orange': [(21, 60)],
                'red': (61, 180)
            },
            'abduction_epaule_droite': {
                'green': (0, 20),
                'orange': [(21, 60)],
                'red': (61, 180)
            },
            # Ajoutez autant de seuils que nécessaire pour d'autres articulations ou mouvements
        }

        try:
            logging.info(f"Analyse de l'image: {image_path}")
            # Charger l'image
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Impossible de charger l'image: {image_path}")
                return '_' + '_'.join(map(str, fields))
            logging.debug("Image chargée avec succès pour l'analyse.")

            image_original = image.copy()
            image = self.preprocess_image(image)
            if image is None:
                logging.error("Échec du prétraitement de l'image.")
                return '_' + '_'.join(map(str, fields))

            # Traitement de l'image avec les deux modèles
            with self.mediapipe_lock:
                logging.info("Traitement de l'image avec Mediapipe Holistic...")
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results_holistic = self.holistic.process(image_rgb)
                logging.info("Traitement de l'image avec Mediapipe Pose...")
                results_pose = self.pose.process(image_rgb)

            # Fusion des résultats
            pose_landmarks = None
            if results_holistic.pose_landmarks:
                pose_landmarks = results_holistic.pose_landmarks
                logging.info("Pose détectée avec le modèle Holistic.")
            elif results_pose.pose_landmarks:
                pose_landmarks = results_pose.pose_landmarks
                logging.info("Pose détectée avec le modèle Pose.")
            else:
                logging.info("Aucune pose détectée avec les deux modèles.")

            # Mettre à jour le champ presence_personne
            presence_personne = 1 if pose_landmarks else 0
            fields[2] = presence_personne  # Mise à jour de la valeur

            if presence_personne:
                self.consecutive_no_detection_frames = 0  # Réinitialiser le compteur
                landmarks = pose_landmarks.landmark
                keypoints = self.extract_keypoints(landmarks)

                # Calcul des angles pour différentes articulations
                angles = {}
                if all(keypoints.get(k) is not None for k in [KeyPoints.SHOULDER_LEFT_ROTATION, KeyPoints.ELBOW_LEFT, KeyPoints.WRIST_LEFT]):
                    angles['flexion_epaule_gauche'] = self.calculate_angle(
                        keypoints[KeyPoints.ELBOW_LEFT],
                        keypoints[KeyPoints.SHOULDER_LEFT_ROTATION],
                        keypoints[KeyPoints.HIP_LEFT] if keypoints.get(KeyPoints.HIP_LEFT) else [keypoints[KeyPoints.SHOULDER_LEFT_ROTATION][0], keypoints[KeyPoints.SHOULDER_LEFT_ROTATION][1]+0.1]
                    )

                if all(keypoints.get(k) is not None for k in [KeyPoints.SHOULDER_RIGHT_ROTATION, KeyPoints.ELBOW_RIGHT, KeyPoints.WRIST_RIGHT]):
                    angles['flexion_epaule_droite'] = self.calculate_angle(
                        keypoints[KeyPoints.ELBOW_RIGHT],
                        keypoints[KeyPoints.SHOULDER_RIGHT_ROTATION],
                        keypoints[KeyPoints.HIP_RIGHT] if keypoints.get(KeyPoints.HIP_RIGHT) else [keypoints[KeyPoints.SHOULDER_RIGHT_ROTATION][0], keypoints[KeyPoints.SHOULDER_RIGHT_ROTATION][1]+0.1]
                    )

                if all(keypoints.get(k) is not None for k in [KeyPoints.NECK_FLEXION, KeyPoints.SHOULDER_LEFT_ROTATION, KeyPoints.SHOULDER_RIGHT_ROTATION]):
                    angles['flexion_cou'] = self.calculate_angle(
                        keypoints[KeyPoints.SHOULDER_LEFT_ROTATION],
                        keypoints[KeyPoints.NECK_FLEXION],
                        keypoints[KeyPoints.SHOULDER_RIGHT_ROTATION]
                    )

                # Classification des angles
                risk_scores = {}
                for angle_name, angle_value in angles.items():
                    if angle_name in thresholds:
                        score = self.classify_angle(angle_value, thresholds[angle_name])
                        risk_scores[angle_name] = score
                        logging.info(f"{angle_name}: Angle={angle_value}, Score={score}")
                    else:
                        logging.warning(f"Aucun seuil défini pour {angle_name}")

                # Mise à jour des champs en fonction des scores calculés
                if 'flexion_cou' in angles:
                    fields[0] = angles['flexion_cou']  # Valeur de l'angle
                    fields[1] = risk_scores.get('flexion_cou', 0)  # Score de risque

                # Détermination de la zone de risque (peut être une combinaison de plusieurs scores)
                risk_zone = max(risk_scores.values()) if risk_scores else 0
                fields[3] = risk_zone

                # Détection des actions techniques
                detected_actions = self.detect_actions_techniques_in_image(image_original)
                num_actions = len(detected_actions)
                fields[4] = num_actions

                # ***** Attribution des Scores Réels *****
                current_time = time.time()
                self.action_history.append((current_time, num_actions))
                self.clean_history(self.action_history, self.repetitivite_window)
                
                # Calcul de la répétitivité
                repetitivite_score = sum(a for (t, a) in self.action_history)
                fields[5] = repetitivite_score
                logging.info(f"Repetitivite_score: {repetitivite_score}")

                # Calcul du maintien de la posture
                posture = keypoints.copy()
                self.posture_history.append((current_time, posture))
                self.clean_history(self.posture_history, self.posture_window)
                maintien_posture_score = self.calculate_posture_score()
                fields[6] = maintien_posture_score

                # Calcul du score de récupération
                recuperation_score = self.calculate_recuperation_score(current_time)
                fields[7] = recuperation_score

                # Mise à jour du dernier temps d'activité
                self.last_activity_time = current_time

                # Calcul du score de préhension
                wrists = {
                    'wrist_left': keypoints.get(KeyPoints.WRIST_LEFT),
                    'wrist_right': keypoints.get(KeyPoints.WRIST_RIGHT)
                }
                self.hand_positions_history.append((current_time, wrists))
                self.clean_history(self.hand_positions_history, self.prehension_window)
                prehension_score = self.calculate_prehension_score()
                fields[8] = prehension_score

                # La détection est stable, pas de changement de complexité
            else:
                self.consecutive_no_detection_frames += 1
                logging.info(f"Aucune détection de pose. Compteur: {self.consecutive_no_detection_frames}")

                if self.consecutive_no_detection_frames >= 5:
                    # Alterner la complexité du modèle entre 1 et 2
                    self.desired_model_complexity = 2 if self.current_model_complexity == 1 else 1
                    logging.info(f"Changement de la complexité du modèle à {self.desired_model_complexity} après {self.consecutive_no_detection_frames} frames sans détection.")
                    with self.mediapipe_lock:
                        self.holistic.close()
                        self.holistic = self.mp_holistic.Holistic(
                            min_detection_confidence=0.3,
                            min_tracking_confidence=0.3,
                            model_complexity=self.desired_model_complexity
                        )
                        self.pose.close()
                        self.pose = self.mp_pose.Pose(
                            static_image_mode=False,
                            model_complexity=self.desired_model_complexity,
                            enable_segmentation=False,
                            min_detection_confidence=0.3,
                            min_tracking_confidence=0.3
                        )
                        self.current_model_complexity = self.desired_model_complexity
                        logging.info(f"Complexité des modèles Mediapipe mise à jour à {self.current_model_complexity}")
                    self.consecutive_no_detection_frames = 0  # Réinitialiser le compteur

        except Exception as e:
            logging.exception("Erreur lors de l'estimation")

        # Construction de la chaîne de résultat avec un underscore initial
        result = '_' + '_'.join(map(str, fields))

        logging.info(f"Résultat de l'analyse: {result}")
        return result

    def clean_history(self, history: deque, window: int) -> None:
        """Nettoie l'historique pour ne conserver que les entrées dans la fenêtre spécifiée."""
        current_time = time.time()
        with self.keypoint_lock:
            while history and current_time - history[0][0] > window:
                history.popleft()

    def calculate_posture_score(self) -> float:
        """Calcule le score de maintien de la posture."""
        try:
            if len(self.posture_history) <= 1:
                logging.info("Maintien_posture_score: 100 (aucune variation)")
                return 100.0

            variances = {}
            for key in [KeyPoints.SHOULDER_LEFT_ROTATION, KeyPoints.SHOULDER_RIGHT_ROTATION, KeyPoints.ELBOW_LEFT, KeyPoints.ELBOW_RIGHT]:
                positions = [p[key] for (t, p) in self.posture_history if p.get(key)]
                if positions:
                    x_vals = [pos[0] for pos in positions]
                    y_vals = [pos[1] for pos in positions]
                    variance = np.var(x_vals) + np.var(y_vals)
                    variances[key] = variance

            if variances:
                avg_variance = np.mean(list(variances.values()))
                maintien_posture_score = max(0, 100 - avg_variance * 1000)
                logging.info(f"Maintien_posture_score: {maintien_posture_score}")
                return round(maintien_posture_score, 2)
            else:
                logging.info("Aucune variance calculable pour le maintien de la posture.")
                return 100.0
        except Exception as e:
            logging.exception("Erreur lors du calcul du maintien de la posture")
            return 0.0

    def calculate_recuperation_score(self, current_time: float) -> float:
        """Calcule le score de récupération."""
        try:
            if self.last_activity_time:
                elapsed = current_time - self.last_activity_time
                if elapsed >= self.recuperation_threshold:
                    recuperation_score = min(100, elapsed)
                    logging.info(f"Recuperation_score: {recuperation_score}")
                    return recuperation_score
                else:
                    logging.info("Recuperation_score: 0 (récupération insuffisante)")
                    return 0.0
            else:
                logging.info("Recuperation_score: 0 (aucune activité enregistrée)")
                return 0.0
        except Exception as e:
            logging.exception("Erreur lors du calcul du score de récupération")
            return 0.0

    def calculate_prehension_score(self) -> float:
        """Calcule le score de préhension."""
        try:
            if len(self.hand_positions_history) <= 1:
                logging.info("Prehension_score: 100 (aucune variation)")
                return 100.0

            stability_scores = []
            for wrist in ['wrist_left', 'wrist_right']:
                positions = [p[wrist] for (t, p) in self.hand_positions_history if p.get(wrist)]
                if positions:
                    x_vals = [pos[0] for pos in positions]
                    y_vals = [pos[1] for pos in positions]
                    variance = np.var(x_vals) + np.var(y_vals)
                    stability_scores.append(variance)

            if stability_scores:
                avg_variance = np.mean(stability_scores)
                prehension_score = max(0, 100 - avg_variance * 1000)
                logging.info(f"Prehension_score: {prehension_score}")
                return round(prehension_score, 2)
            else:
                logging.info("Aucune variance calculable pour la préhension.")
                return 100.0
        except Exception as e:
            logging.exception("Erreur lors du calcul du score de préhension")
            return 0.0

    # =============================================
    # Gestion de la Complexité du Modèle
    # =============================================

    # La méthode adjust_model_complexity n'est plus utilisée dans ce contexte

    # =============================================
    # Fonction Périodique
    # =============================================

    def fct_periodique_1s(self) -> None:
        """Fonction périodique exécutée toutes les secondes pour gérer l'enregistrement et la communication."""
        retries = 0
        while not self.stop_event.is_set():
            try:
                HOST = self.full_ip_concentrateur
                PORT = 50000
                logging.info(f"Tentative de connexion au concentrateur {HOST}:{PORT}...")
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    client_socket.settimeout(10)
                    client_socket.connect((HOST, PORT))
                    logging.info("Connecté au concentrateur.")
                    retries = 0

                    while not self.stop_event.is_set():
                        demande_recording = "yes"  # Cette logique peut être améliorée
                        if demande_recording == "yes":
                            self.recording = "yes"
                            filename = self.record_image()
                            if filename:
                                self.result_analyse = self.estimator(filename)
                                try:
                                    os.remove(filename)
                                    logging.info(f"Fichier {filename} supprimé.")
                                except Exception as e:
                                    logging.exception(f"Erreur lors de la suppression du fichier {filename}")
                                self.mdv_app()
                                now_message = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                                self.app_is_on = "yes"
                                message_emission = f"{self.num_poste}_{self.app_is_on}_{self.recording}_{self.pres_cam}_{self.mdv}_0{self.result_analyse}"
                                logging.info(f"{now_message} : {message_emission}")
                                try:
                                    client_socket.sendall(message_emission.encode())
                                    logging.info("Message envoyé au concentrateur.")
                                except (BrokenPipeError, ConnectionResetError) as e:
                                    logging.exception("Erreur lors de l'envoi des données")
                                    break
                                except Exception as e:
                                    logging.exception("Erreur inattendue lors de l'envoi des données")
                                    break
                        else:
                            self.recording = "no"
                        
                        time.sleep(1)  # Pause d'une seconde
            except (socket.timeout, socket.error) as e:
                logging.exception("Erreur de socket lors de la connexion au concentrateur")
                retries += 1
                if retries > self.max_retries:
                    logging.error("Nombre maximal de tentatives de reconnexion atteint. Arrêt du thread périodique.")
                    break
                logging.info(f"Tentative de reconnexion dans {self.retry_delay} secondes...")
                time.sleep(self.retry_delay)
            except Exception as e:
                logging.exception("Exception lors de la fonction périodique")
                retries += 1
                if retries > self.max_retries:
                    logging.error("Nombre maximal de tentatives de reconnexion atteint. Arrêt du thread périodique.")
                    break
                logging.info(f"Tentative de reconnexion dans {self.retry_delay} secondes...")
                time.sleep(self.retry_delay)

    # =============================================
    # Méthodes pour Démarrer les Threads
    # =============================================

    def start_periodic_thread(self) -> None:
        """Démarre le thread périodique."""
        self.periodic_thread = threading.Thread(target=self.fct_periodique_1s, daemon=True)
        self.periodic_thread.start()
        logging.info("Thread périodique démarré.")

    # =============================================
    # Programme Principal
    # =============================================

    def run(self) -> None:
        """Exécute le programme principal de l'application."""
        try:
            logging.info(f"Début de l'application CameraApplication, Version {__version__}")
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            logging.info("Paramètres définis avec succès.")

            logging.info(f"Vérification de la présence du Concentrateur @{self.full_ip_concentrateur} ...")
            success = self.ping_host(host=self.full_ip_concentrateur)
            if success:
                logging.info(colored("Le concentrateur répond aux pings.", "green"))
            else:
                logging.warning(colored("Le concentrateur ne répond pas aux pings.", "red"))

            logging.info("Initialisation AnywhereUSB...")
            self.init_anyusb()

            logging.info("Vérification de la connexion à la caméra...")
            cam_or_not = self.check_cam()
            if cam_or_not == 0:
                logging.warning("Caméra non détectée")
                self.pres_cam = "no"
                
                while self.check_cam() == 0 and not self.stop_event.is_set():
                    logging.info("Réessai de détection de la caméra dans 2 secondes...")
                    time.sleep(2)
                
                if self.stop_event.is_set():
                    logging.info("Arrêt demandé. Fermeture du programme.")
                    return  # Sortir si l'arrêt a été demandé
                
                try:
                    openni2.initialize()
                    logging.info("OpenNI initialisé")
                    self.dev = openni2.Device.open_any()
                    logging.info("Caméra détectée")
                    self.color_stream = self.dev.create_color_stream()
                    self.color_stream.start()
                    logging.info("Flux vidéo de la caméra démarré")
                    self.pres_cam = "yes"
                except Exception as e:
                    logging.exception("Erreur lors de l'initialisation de la caméra")
                    self.pres_cam = "no"
            else:
                try:
                    openni2.initialize()
                    logging.info("OpenNI initialisé")
                    self.dev = openni2.Device.open_any()
                    logging.info("Caméra détectée")
                    self.color_stream = self.dev.create_color_stream()
                    self.color_stream.start()
                    logging.info("Flux vidéo de la caméra démarré")
                    self.pres_cam = "yes"
                except Exception as e:
                    logging.exception("Erreur lors de la connexion à la caméra")
                    self.pres_cam = "no"
            
            if self.pres_cam == "yes":
                logging.info(colored("********** APPLICATION OPÉRATIONNELLE **********", "green"))
                logging.info(colored("Échanges en cours avec le concentrateur", "green"))
                
                self.start_periodic_thread()
                
                try:
                    while not self.stop_event.is_set():
                        time.sleep(10)
                except KeyboardInterrupt:
                    logging.info("Interrompu par l'utilisateur. Fermeture...")
                    self.sortie_programme()
                
                logging.info("Fin du programme principal")
                self.sortie_programme()
            else:
                logging.error("Caméra non initialisée. Arrêt du programme.")
                self.sortie_programme()
        except Exception as e:
            logging.exception("Exception générale dans le programme principal")
            self.sortie_programme()

# =============================================
# Lancement de l'Application
# =============================================

if __name__ == "__main__":
    app = CameraApplication()
    app.run()

# =============================================
# FIN PROGRAMME PRINCIPAL
# =============================================