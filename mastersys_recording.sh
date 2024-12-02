#!/bin/bash

# Variables globales
SCRIPT_NAME="$(basename "$0")"
CHEMIN="/home/pc-camera/Bureau/Cameras/03_Code_MiniPC"
LOGFILE="${CHEMIN}/mastersys_recording.log"

# Fichiers PID
pid_file_recording="${CHEMIN}/mastersys_recording.pid"
pid_file_monitoring="${CHEMIN}/mastersys_monitoring.pid"

# Fonction de journalisation centralisée
log_message() {
    local message="$1"
    local function_name="${FUNCNAME[1]}"
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "${timestamp} - ${SCRIPT_NAME} - ${function_name} [PID $$]: ${message}" >> "$LOGFILE"
}

ensure_processes_killed() {
    log_message "Vérification et arrêt des processus existants..."

    # Tuer tous les processus recording.py
    if pgrep -f "recording.py" > /dev/null; then
        log_message "Arrêt des processus recording.py en cours..."
        pkill -f "recording.py"
    fi

    # Tuer le script de surveillance s'il est en cours d'exécution
    if [ -f "$pid_file_monitoring" ]; then
        pid_monitoring=$(cat "$pid_file_monitoring")
        if kill -0 "$pid_monitoring" > /dev/null 2>&1; then
            log_message "Arrêt du script de surveillance en cours avec PID $pid_monitoring..."
            kill "$pid_monitoring"
        fi
        rm -f "$pid_file_monitoring"
    fi

    # Supprimer les fichiers PID existants
    rm -f "$pid_file_recording" "$pid_file_monitoring"

    log_message "Tous les processus existants ont été arrêtés."
}

start_programme() {
    cd "$CHEMIN" || exit 1

    # Démarrer le programme
    log_message "Démarrage du programme recording.py..."
    python3 recording.py &
    # Enregistrer le PID dans le fichier
    pid_recording=$!
    echo "$pid_recording" > "$pid_file_recording"
    log_message "pid_recording : $pid_recording"
}

stop_programme() {
    log_message "Arrêt de tous les processus recording.py et du script de surveillance..."

    # Tuer tous les processus recording.py
    pkill -f "recording.py"

    # Supprimer le fichier PID du programme
    rm -f "$pid_file_recording"

    # Tuer le script de surveillance
    if [ -f "$pid_file_monitoring" ]; then
        pid_monitoring=$(cat "$pid_file_monitoring")
        log_message "Arrêt du script de surveillance avec PID $pid_monitoring..."
        kill "$pid_monitoring"
        rm -f "$pid_file_monitoring"
    else
        log_message "Aucun script de surveillance en cours d'exécution."
    fi

    log_message "Tous les processus arrêtés."
}

check_cron_job() {
    log_message "Vérification de la présence d'une tâche CRON associée au script..."

    # Obtenir le crontab de l'utilisateur
    cron_jobs=$(crontab -l 2>/dev/null)

    # Vérifier si le crontab a été récupéré avec succès
    if [ $? -ne 0 ]; then
        log_message "Impossible de récupérer le crontab de l'utilisateur. Peut-être qu'aucune tâche CRON n'est définie."
        return
    fi

    # Vérifier si le script est présent dans le crontab
    if echo "$cron_jobs" | grep -q "$SCRIPT_NAME"; then
        log_message "Une tâche CRON est associée au script."
    else
        log_message "Aucune tâche CRON associée au script n'a été trouvée."
    fi
}

#-----------------------------------------------------#
#                       MAIN                          #
#-----------------------------------------------------#

case "$1" in
    start)
        # S'assurer que les processus correspondants sont arrêtés avant de démarrer
        ensure_processes_killed

        log_message "########## DÉMARRAGE DU SCRIPT DE SURVEILLANCE ##########"

        # Enregistrer le PID du script de surveillance
        echo "$$" > "$pid_file_monitoring"

        # Démarrer le programme initialement
        start_programme

        # Vérifier la présence d'une tâche CRON associée au script
        check_cron_job

        # Surveiller l'existence du processus et relancer si nécessaire
        while true; do
            sleep 30  # Attendre 30 secondes avant de vérifier à nouveau
            if pgrep -f "recording.py" > /dev/null; then
                log_message "recording.py est en cours d'exécution."
            else
                log_message "recording.py s'est arrêté. Redémarrage..."
                start_programme
            fi
        done
        ;;
    stop)
        stop_programme
        ;;
    *)
        log_message "Action non reconnue. Utilisation: $0 {start|stop}"
        exit 1
        ;;
esac
