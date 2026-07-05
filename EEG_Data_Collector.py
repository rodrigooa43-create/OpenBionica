"""
EEG_Data_Collector.py — Lime Edition (com suporte ao módulo de expansão 16 canais)
================================================================================
Aplicação 100% Python (PySide6 + pyqtgraph + scipy + numpy) para coleta e análise
de sinais EEG em tempo real. Inspirada em interfaces clássicas de aquisição EEG, com identidade
própria (tema dark + verde-limao).

Recursos
--------
- Captura 8 canais (250 Hz) ou 16 canais (com módulo de expansão +8 canais, 125 Hz efetivo)
- Auto-detecção de expansão:
    * CSV de playback com >= 9+1+3 colunas -> ativa modo 16 canais
    * Botão "Detectar expansão" envia comandos Cyton e questiona status
    * Toggle manual sempre disponivel
- Modos: Hardware, Simulação, Playback
- 9 abas:
    1. Conexão            — porta COM, baud, modo, MODULO DE EXPANSAO, log
    2. Tempo Real         — todos os canais + acelerômetro + markers
    3. Análises           — FFT, bandas, statisticas
    4. Topografia         — Head Plot 10-20 (8 ou 16 eletrodos), Focus, EMG
    5. Espectrograma      — heatmap freq x tempo por canal
    6. Filtros & Canais   — notch + bandpass + toggles
    7. Hardware           — grid por canal (gain/mode/SRB) + comandos
    8. Rede & Eventos     — UDP JSON + injecao de markers
    9. Histórico          — overlay temporal

Caminhos (em Documents/EEG_Coletor/)
- Logos:        %USERPROFILE%/Documents/EEG_Coletor/Logo_*.png|.jpg
- Assets externos: %USERPROFILE%/Documents/EEG_Coletor/OpenBCI_GUI-master/  (referencia)
- Sessões CSV:  %USERPROFILE%/Documents/EEG_Coletor/sessions/

Uso:
    python EEG_Data_Collector.py
"""

import csv
import difflib
import hashlib
import json
import logging
import logging.handlers
import math
import os
import re
import secrets
import socket
import sys
import time
import traceback
import unicodedata
from collections import deque
from datetime import datetime

import numpy as np
# Garante que o pyqtgraph use o MESMO binding Qt do app (PySide6, LGPL)
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
import pyqtgraph as pg
import serial
import serial.tools.list_ports
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QThread, QTimer, Signal
from scipy import signal as scipy_signal
from scipy.fft import rfft, rfftfreq

# ============================================================
# Dependências opcionais — detecção SEM importar (lazy import)
# ------------------------------------------------------------
# Importar mne / matplotlib / scikit-learn / pyedflib logo no
# arranque custava cerca de 3 segundos. Aqui apenas verificamos se
# o pacote ESTÁ instalado (find_spec é rápido e NÃO executa o
# módulo). O import de verdade acontece dentro da função que usa
# cada recurso (export, ICA, PDF, LSL...), deixando a abertura do
# programa muito mais rápida.
# ============================================================
import importlib.util as _ilu


def _module_available(name):
    """True se o pacote está instalado, sem importá-lo (rápido)."""
    try:
        return _ilu.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


# Sentinelas globais: o import real é feito localmente, sob demanda,
# dentro de cada função que usa o recurso.
pylsl = None
pyedflib = None
mne = None
plt = None

HAS_LSL       = _module_available("pylsl")
HAS_EDF       = _module_available("pyedflib")
HAS_MNE       = _module_available("mne")
HAS_MPL       = _module_available("matplotlib")
HAS_REPORTLAB = _module_available("reportlab")
# scikit-learn deixou de ser necessário: o BCI Trainer faz CSP via
# scipy.linalg.eigh e LDA de Fisher manual. Flag mantida por
# compatibilidade (sempre False — nada no app importa sklearn).
HAS_SKLEARN   = False


# ============================================================
# Caminhos
# ============================================================
_USER_DIR = os.path.expanduser("~")
_DOC_CANDS = [
    os.path.join(_USER_DIR, "Documents", "EEG_Coletor"),
    os.path.join(_USER_DIR, "OneDrive", "Documentos", "EEG_Coletor"),
    os.path.join(_USER_DIR, "OneDrive", "Documents", "EEG_Coletor"),
]
DOC_DIR = next((d for d in _DOC_CANDS if os.path.isdir(d)), _DOC_CANDS[0])
LOGO_UFES_PATH    = os.path.join(DOC_DIR, "Logo_Ufes.png")
LOGO_BIONICA_PATH = os.path.join(DOC_DIR, "Logo_BionicaLab.jpg")
EXTERNAL_GUI_DIR  = os.path.join(DOC_DIR, "OpenBCI_GUI-master")  # pasta de assets externos (mantida por compatibilidade)
CONFIG_PATH       = os.path.join(DOC_DIR, "config.json")
os.makedirs(DOC_DIR, exist_ok=True)

# Diretório padrão de salvamento: <pasta do .py>/sessions/
# (portátil — vai junto com o script). Usuário pode mudar nas Configurações.
if getattr(sys, "frozen", False):
    # Executável PyInstaller (.exe): usar a pasta REAL do executável e
    # NÃO a pasta temporária de extração (sys._MEIPASS). Assim as
    # gravações em sessions/ ficam ao lado do .exe e persistem.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        SCRIPT_DIR = os.getcwd()
DEFAULT_SAVE_DIRECTORY = os.path.join(SCRIPT_DIR, "sessions")
# alias retrocompativel — leitura inicial usa o default; o valor "real" fica em AppConfig
SAVE_DIRECTORY = DEFAULT_SAVE_DIRECTORY
os.makedirs(SAVE_DIRECTORY, exist_ok=True)

# ------------------------------------------------------------
# Fallback portátil de assets (logos)
# Por padrão os logos são lidos de Documents/EEG_Coletor. Quando o
# app roda a partir da própria pasta (cópia distribuível) ou de um
# executável PyInstaller, procuramos os logos também ao lado do
# script/exe (e em sys._MEIPASS no .exe). Não altera o comportamento
# original — só adiciona locais de busca extras.
# ------------------------------------------------------------
_ASSET_DIRS = [DOC_DIR, SCRIPT_DIR]
_MEIPASS_DIR = getattr(sys, "_MEIPASS", None)
if _MEIPASS_DIR:
    _ASSET_DIRS.append(_MEIPASS_DIR)


def _resolve_asset(filename, default):
    for _base in _ASSET_DIRS:
        _cand = os.path.join(_base, filename)
        if os.path.exists(_cand):
            return _cand
    return default


LOGO_UFES_PATH    = _resolve_asset("Logo_Ufes.png", LOGO_UFES_PATH)
LOGO_BIONICA_PATH = _resolve_asset("Logo_BionicaLab.jpg", LOGO_BIONICA_PATH)


# ============================================================
# Sistema 10-20 completo (com Fz, Cz, Pz, CPz, FCz, etc.)
# Posicoes normalizadas (-1..1, y+ = frente)
# ============================================================
ALL_ELECTRODES = {
    # Linha frontal
    "Fp1": (-0.25,  0.85), "Fpz": ( 0.00,  0.90), "Fp2": ( 0.25,  0.85),
    "AF7": (-0.55,  0.72), "AF3": (-0.30,  0.72), "AFz": ( 0.00,  0.74),
    "AF4": ( 0.30,  0.72), "AF8": ( 0.55,  0.72),
    "F7":  (-0.75,  0.55), "F5":  (-0.55,  0.55), "F3":  (-0.35,  0.55),
    "F1":  (-0.15,  0.55), "Fz":  ( 0.00,  0.55), "F2":  ( 0.15,  0.55),
    "F4":  ( 0.35,  0.55), "F6":  ( 0.55,  0.55), "F8":  ( 0.75,  0.55),
    # Linha fronto-central
    "FT7": (-0.85,  0.30), "FC5": (-0.62,  0.30), "FC3": (-0.40,  0.30),
    "FC1": (-0.18,  0.30), "FCz": ( 0.00,  0.30), "FC2": ( 0.18,  0.30),
    "FC4": ( 0.40,  0.30), "FC6": ( 0.62,  0.30), "FT8": ( 0.85,  0.30),
    # Linha central
    "T7":  (-0.95,  0.00), "C5":  (-0.70,  0.00), "C3":  (-0.45,  0.00),
    "C1":  (-0.22,  0.00), "Cz":  ( 0.00,  0.00), "C2":  ( 0.22,  0.00),
    "C4":  ( 0.45,  0.00), "C6":  ( 0.70,  0.00), "T8":  ( 0.95,  0.00),
    # Linha centro-parietal
    "TP7": (-0.85, -0.30), "CP5": (-0.62, -0.30), "CP3": (-0.40, -0.30),
    "CP1": (-0.18, -0.30), "CPz": ( 0.00, -0.30), "CP2": ( 0.18, -0.30),
    "CP4": ( 0.40, -0.30), "CP6": ( 0.62, -0.30), "TP8": ( 0.85, -0.30),
    # Linha parietal
    "P7":  (-0.75, -0.55), "P5":  (-0.55, -0.55), "P3":  (-0.35, -0.55),
    "P1":  (-0.15, -0.55), "Pz":  ( 0.00, -0.55), "P2":  ( 0.15, -0.55),
    "P4":  ( 0.35, -0.55), "P6":  ( 0.55, -0.55), "P8":  ( 0.75, -0.55),
    # Linha parieto-occipital
    "PO7": (-0.55, -0.72), "PO3": (-0.30, -0.72), "POz": ( 0.00, -0.74),
    "PO4": ( 0.30, -0.72), "PO8": ( 0.55, -0.72),
    # Occipital
    "O1":  (-0.25, -0.85), "Oz":  ( 0.00, -0.90), "O2":  ( 0.25, -0.85),
}
ELECTRODE_NAMES = sorted(ALL_ELECTRODES.keys())

# Mapeamento padrão CH -> eletrodo (placa base + até 7 módulos de expansão = 64ch)
# Os primeiros 16 mantêm a montagem clássica OpenBCI/BCI; os demais seguem
# uma sequência canônica do sistema 10-20 estendido (montagem 64 ch típica).
DEFAULT_MAPPING = [
    # CH1-8  — placa base (montagem essencial para MI/SSVEP)
    "Fp1", "Fp2", "C3",  "C4",  "P7",  "P8",  "O1",  "O2",
    # CH9-16 — 1ª expansão (frontal + temporal + parietal mediana)
    "F7",  "F8",  "F3",  "F4",  "T7",  "T8",  "P3",  "P4",
    # CH17-24 — 2ª expansão (linha medial + fronto-central)
    "Fz",  "Cz",  "Pz",  "Oz",  "FCz", "CPz", "FC3", "FC4",
    # CH25-32 — 3ª expansão (centro-parietal e temporal)
    "CP3", "CP4", "FC1", "FC2", "CP1", "CP2", "C1",  "C2",
    # CH33-40 — 4ª expansão (parietal e frontal complementar)
    "P1",  "P2",  "F1",  "F2",  "AF3", "AF4", "PO3", "PO4",
    # CH41-48 — 5ª expansão (linhas laterais)
    "F5",  "F6",  "C5",  "C6",  "P5",  "P6",  "FT7", "FT8",
    # CH49-56 — 6ª expansão (temporo-parietal e frontal extrema)
    "TP7", "TP8", "FC5", "FC6", "CP5", "CP6", "AF7", "AF8",
    # CH57-64 — 7ª expansão (occipital e antero-frontal)
    "PO7", "PO8", "POz", "Fpz", "AFz", "F8",  "T8",  "P8",
]
# Garantia: sempre tem pelo menos 64 itens (pad com placeholders se preciso).
# MAX_CHANNELS é definido mais abaixo no arquivo (= 64), então usamos 64 como
# constante local aqui. O slicing final em [:MAX_CHANNELS] é refeito quando
# MAX_CHANNELS aparece (logo abaixo).
_MAPPING_TARGET = 64
if len(DEFAULT_MAPPING) < _MAPPING_TARGET:
    DEFAULT_MAPPING = DEFAULT_MAPPING + [f"E{i+1}" for i in range(
        len(DEFAULT_MAPPING), _MAPPING_TARGET)]
DEFAULT_MAPPING = DEFAULT_MAPPING[:_MAPPING_TARGET]


# ============================================================
# Temas (paletas de cores)
# ============================================================
THEMES = {
    "Lime (verde-limao)": {
        "background":  "#0d0d0d", "surface":     "#1a1a1a",
        "surface_alt": "#252525", "border":      "#333333",
        "text":        "#e0e0e0", "text_dim":    "#909090",
        "accent":      "#a8ff00", "accent_dim":  "#5a8a00",
        "error":       "#ff3355", "warning":     "#ffaa00",
        "expansion":   "#66bbff",
        "table_bg":    "#1a1a1a", "table_alt":   "#252525",
    },
    "Claro Clinico": {
        # Tema claro clinico (Proposta 2): superficies brancas, acento
        # verde-azulado profissional, cinzas neutros e bordas sutis.
        "background":  "#f7f9fb", "surface":     "#ffffff",
        "surface_alt": "#f1f4f8", "border":      "#dbe1ea",
        "text":        "#1a2230", "text_dim":    "#5b6473",
        "accent":      "#0f9d75", "accent_dim":  "#0c7f5f",
        "error":       "#d4364f", "warning":     "#c77700",
        "expansion":   "#2563c9",
        "table_bg":    "#ffffff", "table_alt":   "#eef4f1",
    },
    "Claro (white)": {
        "background":  "#fafafa", "surface":     "#ffffff",
        "surface_alt": "#f0f0f0", "border":      "#cccccc",
        "text":        "#1a1a1a", "text_dim":    "#666666",
        "accent":      "#0066cc", "accent_dim":  "#0099ff",
        "error":       "#cc0033", "warning":     "#cc7700",
        "expansion":   "#9933cc",
        "table_bg":    "#ffffff", "table_alt":   "#eef2f6",
    },
    "Escuro Puro (black)": {
        "background":  "#000000", "surface":     "#0a0a0a",
        "surface_alt": "#181818", "border":      "#2a2a2a",
        "text":        "#ffffff", "text_dim":    "#888888",
        "accent":      "#ffffff", "accent_dim":  "#aaaaaa",
        "error":       "#ff4466", "warning":     "#ffbb22",
        "expansion":   "#99ccff",
        "table_bg":    "#0a0a0a", "table_alt":   "#181818",
    },
    "Sistema (default)": {
        "background":  "#2b2b2b", "surface":     "#3c3c3c",
        "surface_alt": "#454545", "border":      "#555555",
        "text":        "#f0f0f0", "text_dim":    "#aaaaaa",
        "accent":      "#5599ff", "accent_dim":  "#4477cc",
        "error":       "#ee5566", "warning":     "#eebb33",
        "expansion":   "#88ddff",
        "table_bg":    "#3c3c3c", "table_alt":   "#454545",
    },
}


# ============================================================
# I18N — Internacionalização (pt / en / es)
# ============================================================
class I18N:
    """Sistema simples de tradução: chave = string em pt-BR, valor = string traduzida.

    Strings ausentes do mapa retornam o original (pt-BR) — fallback seguro.
    A mudança real de idioma requer reiniciar o app para que todas as strings
    sejam re-construídas (a maioria está hardcoded em pt). Alguns títulos e
    botões principais são re-aplicados em runtime via _apply_language().
    """

    LANGUAGES = {
        "pt": "Português (BR)",
        "en": "English (US)",
        "es": "Español",
    }

    # Dicionários ENGLISH — chaves em pt-BR
    _en = {
        # ===== Grupos top-level =====
        "Configurar": "Setup",
        "Visualizar": "Visualize",
        "Analisar":  "Analyze",
        "Sistema":   "System",
        # ===== Sub-abas - Configurar =====
        "Voluntários":      "Volunteers",
        "Conexão":          "Connection",
        "Filtros e Canais": "Filters and Channels",
        "Hardware":         "Hardware",
        "Calibração":       "Calibration",
        # ===== Sub-abas - Visualizar =====
        "Tempo Real":         "Real Time",
        "Topografia":         "Topography",
        "Espectrograma":      "Spectrogram",
        "EMG / Músculos":     "EMG / Muscles",
        "Bio (EMG/ECG/EoG)":  "Bio (EMG/ECG/EoG)",
        "Histórico":          "History",
        "Layout Custom":      "Custom Layout",
        # ===== Sub-abas - Analisar =====
        "Análises":      "Analyses",
        "Offline":       "Offline",
        "ERP":           "ERP",
        "Conectividade": "Connectivity",
        "ERS/ERD":       "ERS/ERD",
        "Focus / SSVEP":    "Focus / SSVEP",
        "EMG Joystick":     "EMG Joystick",
        "BCI Trainer (MI)": "BCI Trainer (MI)",
        # ===== Sub-abas - Sistema =====
        "Rede e Eventos": "Network and Events",
        "Configurações":  "Settings",
        # ===== Header =====
        "DESCONECTADO": "DISCONNECTED",
        "CONECTADO":    "CONNECTED",
        "GRAVANDO":     "RECORDING",
        "nenhum":       "none",
        "Pronto.":      "Ready.",
        "Amostras: 0":  "Samples: 0",
        # ===== Menus / Ajuda =====
        "Ajuda":                    "Help",
        "Sobre o aplicativo...":    "About this application...",
        "Atalhos de teclado...":    "Keyboard shortcuts...",
        "Abrir pasta de sessões":   "Open sessions folder",
        "Idioma alterado":          "Language changed",
        # ===== Botões comuns =====
        "Conectar":     "Connect",
        "Desconectar":  "Disconnect",
        "Gravar":       "Record",
        "Parar":        "Stop",
        "Procurar...":  "Browse...",
        "Cancelar":     "Cancel",
        "Cadastrar":    "Register",
        "Deletar":      "Delete",
        "Salvar":       "Save",
        "Aplicar":      "Apply",
        "Atualizar":    "Refresh",
        "Aplicar modo": "Apply mode",
        "Restaurar Padrão": "Restore Default",
        "Aplicar a todos":  "Apply to all",
        "Reset contadores": "Reset counters",
        "Reset":            "Reset",
        "Enviar":           "Send",
        "Padrão":           "Default",
        "Abrir":            "Open",
        "Copiar":           "Copy",
        "+ Novo":           "+ New",
        "Sair":             "Exit",
        # ===== Launcher =====
        "Voluntário Ativo":                 "Active Volunteer",
        "Sessões Recentes":                 "Recent Sessions",
        "Abrir CSV manualmente...":         "Open CSV manually...",
        "O que você deseja fazer hoje?":    "What do you want to do today?",
        "Escolha um fluxo de trabalho para começar":
            "Choose a workflow to get started",
        "Nova Coleta":                      "New Recording",
        "Analisar Dados":                   "Analyze Data",
        "Aplicações BCI":                   "BCI Applications",
        "Modo Simulação":                   "Simulation Mode",
        "Conectar ao hardware e gravar uma sessão em tempo real":
            "Connect to hardware and record a session in real time",
        "Abrir um CSV e explorar análises offline (FFT, ERP, ERS/ERD)":
            "Open a CSV and explore offline analyses (FFT, ERP, ERS/ERD)",
        "Biofeedback interativo: Focus, EMG Joystick, SSVEP":
            "Interactive biofeedback: Focus, EMG Joystick, SSVEP",
        "Gerar dados sintéticos para testar a interface sem hardware":
            "Generate synthetic data to test the interface without hardware",
        "Pré-Flight Check":                 "Pre-Flight Check",
        "Configure o hardware antes de iniciar a coleta. Estes parâmetros são aplicados ao iniciar.":
            "Configure hardware before starting the recording. These parameters are applied at startup.",
        "Configuração de Hardware":         "Hardware Configuration",
        "Tipo de Aquisição":                "Acquisition Type",
        "Apenas EEG":                       "EEG only",
        "Apenas EMG":                       "EMG only",
        "Apenas ECG":                       "ECG only",
        "Apenas EoG":                       "EoG only",
        "Híbrido (multimodal)":             "Hybrid (multimodal)",
        "Módulo de Expansão (16 Canais)":   "Expansion Module (16 Channels)",
        "Marque se a placa de expansão (8 canais extras) está conectada.":
            "Check if the expansion board (8 extra channels) is connected.",
        "16 canais marcados como EEG. Filtro 0.5-70 Hz + notch.":
            "16 channels marked as EEG. Filter 0.5-70 Hz + notch.",
        "16 canais marcados como EMG. Filtro 20-Nyquist + notch.":
            "16 channels marked as EMG. Filter 20-Nyquist + notch.",
        "16 canais marcados como ECG. Filtro 0.5-100 Hz + notch.":
            "16 channels marked as ECG. Filter 0.5-100 Hz + notch.",
        "Mistura: 1-8 EEG / 9-12 EMG / 13-14 ECG / 15-16 EoG.":
            "Mix: 1-8 EEG / 9-12 EMG / 13-14 ECG / 15-16 EoG.",
        "Porta:":                           "Port:",
        "Modo:":                            "Mode:",
        "Atualizar lista de portas COM":    "Refresh COM ports list",
        "Setup pronto:":                    "Ready setup:",
        "Abrir o aplicativo direto na aba Configurações":
            "Open the application directly on the Settings tab",
        "Modo de Aquisição — Visibilidade de Abas":
            "Acquisition Mode — Tabs Visibility",
        "Esconde abas que não fazem sentido para o modo escolhido (ex.: Topografia/ERP são EEG-only).":
            "Hides tabs that don't make sense for the chosen mode (e.g.: Topography/ERP are EEG-only).",
        # ===== GroupBoxes comuns =====
        "Tema (paleta de cores)":           "Theme (color palette)",
        "Idioma / Language / Idioma":       "Language / Idioma",
        "Selecione:":                       "Select:",
        "Editor de Tema Personalizado":     "Custom Theme Editor",
        "Filtro Notch (rejeição de banda)": "Notch Filter (band-stop)",
        "Filtro Bandpass (Butterworth, ordem 4)":
            "Bandpass Filter (Butterworth, order 4)",
        "Ativado":                          "Enabled",
        "Frequência (Hz):":                 "Frequency (Hz):",
        "Corte inferior (Hz):":             "Lower cutoff (Hz):",
        "Corte superior (Hz):":             "Upper cutoff (Hz):",
        "Presets:":                         "Presets:",
        "Canais — Ativação e Tipo de Sinal (multimodal: EEG / EMG / ECG / EoG)":
            "Channels — Activation and Signal Type (multimodal: EEG / EMG / ECG / EoG)",
        "Sessão e Arquivos":                "Session and Files",
        "Caminhos e Auditoria":             "Paths and Audit",
        "Caminhos do Sistema (editáveis)":  "System Paths (editable)",
        "Log de Auditoria (events.jsonl)":  "Audit Log (events.jsonl)",
        "Pasta de Salvamento das Sessões":  "Sessions Save Folder",
        "Exportar Sessão (escolher .csv ou pasta de sessão)":
            "Export Session (choose .csv or session folder)",
        "Aplicar agora":                    "Apply now",
        "Salvar como novo tema":            "Save as new theme",
        "Deletar tema":                     "Delete theme",
        # ===== EMG / ECG / EoG / Focus / Joystick =====
        "Configuração do Envelope EMG":     "EMG Envelope Configuration",
        "Método:":                          "Method:",
        "Janela (ms):":                     "Window (ms):",
        "Threshold global:":                "Global threshold:",
        "Envelope EMG — últimos 10 s":      "EMG envelope — last 10 s",
        "0 canais EMG ativos":              "0 EMG channels active",
        "Canal ECG:":                       "ECG channel:",
        "Sinal ECG (filtrado 5-15 Hz)":     "ECG signal (filtered 5-15 Hz)",
        "MWA (integral) + threshold":       "MWA (integral) + threshold",
        "Tacograma — intervalos RR (ms)":   "Tachogram — RR intervals (ms)",
        "Poincaré — RR(n) × RR(n+1)":       "Poincaré — RR(n) × RR(n+1)",
        "Canal HEoG (horizontal):":         "HEoG channel (horizontal):",
        "Canal VEoG (vertical):":           "VEoG channel (vertical):",
        "Threshold (µV):":                  "Threshold (µV):",
        "Centro":                           "Center",
        "Cima":                             "Up",
        "Baixo":                            "Down",
        "Esquerda":                         "Left",
        "Direita":                          "Right",
        "piscadas detectadas":              "blinks detected",
        "piscadas/min":                     "blinks/min",
        "Canal EEG:":                       "EEG channel:",
        "Freq SSVEP alvo (Hz):":            "Target SSVEP freq (Hz):",
        "Definir baseline (5s)":            "Set baseline (5s)",
        "Estado":                           "State",
        "FOCADO":                           "FOCUSED",
        "RELAXADO":                         "RELAXED",
        "NORMAL":                           "NORMAL",
        "(sem baseline)":                   "(no baseline)",
        "Mapeamento dos 4 canais EMG → eixos":
            "4-Channel EMG → axes mapping",
        "Direção":                          "Direction",
        "Canal EMG":                        "EMG channel",
        "Envelope atual":                   "Current envelope",
        "Max calibrado":                    "Calibrated max",
        "Calibrar (3s contração)":          "Calibrate (3s contraction)",
        "Calibrar":                         "Calibrate",
        "Dead zone:":                       "Dead zone:",
        "Smoothing (frames):":              "Smoothing (frames):",
        "Sem canal":                        "No channel",
        "Selecione um canal EMG primeiro.": "Select an EMG channel first.",
        # ===== Conexão / Hardware =====
        "Modo de Aquisição":                "Acquisition Mode",
        "Hardware (porta COM real)":        "Hardware (real COM port)",
        "Simulação (sinal sintético)":      "Simulation (synthetic signal)",
        "Playback (replay de CSV)":         "Playback (CSV replay)",
        "Arquivo de playback:":             "Playback file:",
        # ===== Voluntário =====
        "Novo voluntário":                  "New volunteer",
        "Editar voluntário":                "Edit volunteer",
        "Atualizar lista":                  "Refresh list",
        "Ativo: (nenhum)":                  "Active: (none)",
        "Histórico de sessões do voluntário selecionado":
            "Session history of selected volunteer",
        # ===== Banner Bio =====
        "Modalidades bioelétricas — placa multimodal Bionica Lab. Configure o tipo de cada canal em <b>Filtros e Canais → Tipo de sinal</b>.":
            "Bioelectric modalities — Bionica Lab multimodal board. Configure each channel type in <b>Filters and Channels → Signal type</b>.",
        # ===== Status / Logs =====
        "Caminho copiado":                  "Path copied",
        # ===== Settings =====
        "Tema:":   "Theme:",
        "Idioma:": "Language:",
        "Reinicie o app após trocar de idioma para aplicar em todas as telas.":
            "Restart the app after changing language to apply on all screens.",
        # ===== Tipos de sinal =====
        "EEG": "EEG",
        "EMG": "EMG",
        "ECG": "ECG",
        "EoG": "EoG",
        "off": "off",
    }

    # Dicionários ESPAÑOL — chaves em pt-BR
    _es = {
        # ===== Grupos top-level =====
        "Configurar": "Configurar",
        "Visualizar": "Visualizar",
        "Analisar":  "Analizar",
        "Sistema":   "Sistema",
        # ===== Sub-abas - Configurar =====
        "Voluntários":      "Voluntarios",
        "Conexão":          "Conexión",
        "Filtros e Canais": "Filtros y Canales",
        "Hardware":         "Hardware",
        "Calibração":       "Calibración",
        # ===== Sub-abas - Visualizar =====
        "Tempo Real":         "Tiempo Real",
        "Topografia":         "Topografía",
        "Espectrograma":      "Espectrograma",
        "EMG / Músculos":     "EMG / Músculos",
        "Bio (EMG/ECG/EoG)":  "Bio (EMG/ECG/EoG)",
        "Histórico":          "Historial",
        "Layout Custom":      "Diseño Personalizado",
        # ===== Sub-abas - Analisar =====
        "Análises":      "Análisis",
        "Offline":       "Sin Conexión",
        "ERP":           "ERP",
        "Conectividade": "Conectividad",
        "ERS/ERD":       "ERS/ERD",
        "Focus / SSVEP":    "Foco / SSVEP",
        "EMG Joystick":     "Joystick EMG",
        "BCI Trainer (MI)": "Entrenador BCI (MI)",
        # ===== Sub-abas - Sistema =====
        "Rede e Eventos": "Red y Eventos",
        "Configurações":  "Ajustes",
        # ===== Header =====
        "DESCONECTADO": "DESCONECTADO",
        "CONECTADO":    "CONECTADO",
        "GRAVANDO":     "GRABANDO",
        "nenhum":       "ninguno",
        "Pronto.":      "Listo.",
        "Amostras: 0":  "Muestras: 0",
        # ===== Menus / Ajuda =====
        "Ajuda":                    "Ayuda",
        "Sobre o aplicativo...":    "Acerca de la aplicación...",
        "Atalhos de teclado...":    "Atajos de teclado...",
        "Abrir pasta de sessões":   "Abrir carpeta de sesiones",
        "Idioma alterado":          "Idioma cambiado",
        # ===== Botões comuns =====
        "Conectar":     "Conectar",
        "Desconectar":  "Desconectar",
        "Gravar":       "Grabar",
        "Parar":        "Detener",
        "Procurar...":  "Buscar...",
        "Cancelar":     "Cancelar",
        "Cadastrar":    "Registrar",
        "Deletar":      "Eliminar",
        "Salvar":       "Guardar",
        "Aplicar":      "Aplicar",
        "Atualizar":    "Actualizar",
        "Aplicar modo": "Aplicar modo",
        "Restaurar Padrão": "Restaurar Predeterminado",
        "Aplicar a todos":  "Aplicar a todos",
        "Reset contadores": "Reiniciar contadores",
        "Reset":            "Reiniciar",
        "Enviar":           "Enviar",
        "Padrão":           "Predeterm.",
        "Abrir":            "Abrir",
        "Copiar":           "Copiar",
        "+ Novo":           "+ Nuevo",
        "Sair":             "Salir",
        # ===== Launcher =====
        "Voluntário Ativo":                 "Voluntario Activo",
        "Sessões Recentes":                 "Sesiones Recientes",
        "Abrir CSV manualmente...":         "Abrir CSV manualmente...",
        "O que você deseja fazer hoje?":    "¿Qué deseas hacer hoy?",
        "Escolha um fluxo de trabalho para começar":
            "Elige un flujo de trabajo para comenzar",
        "Nova Coleta":                      "Nueva Captura",
        "Analisar Dados":                   "Analizar Datos",
        "Aplicações BCI":                   "Aplicaciones BCI",
        "Modo Simulação":                   "Modo Simulación",
        "Conectar ao hardware e gravar uma sessão em tempo real":
            "Conectar al hardware y grabar una sesión en tiempo real",
        "Abrir um CSV e explorar análises offline (FFT, ERP, ERS/ERD)":
            "Abrir un CSV y explorar análisis sin conexión (FFT, ERP, ERS/ERD)",
        "Biofeedback interativo: Focus, EMG Joystick, SSVEP":
            "Biofeedback interactivo: Foco, Joystick EMG, SSVEP",
        "Gerar dados sintéticos para testar a interface sem hardware":
            "Generar datos sintéticos para probar la interfaz sin hardware",
        "Pré-Flight Check":                 "Verificación Previa",
        "Configure o hardware antes de iniciar a coleta. Estes parâmetros são aplicados ao iniciar.":
            "Configura el hardware antes de iniciar la captura. Estos parámetros se aplican al iniciar.",
        "Configuração de Hardware":         "Configuración de Hardware",
        "Tipo de Aquisição":                "Tipo de Adquisición",
        "Apenas EEG":                       "Solo EEG",
        "Apenas EMG":                       "Solo EMG",
        "Apenas ECG":                       "Solo ECG",
        "Apenas EoG":                       "Solo EoG",
        "Híbrido (multimodal)":             "Híbrido (multimodal)",
        "Módulo de Expansão (16 Canais)":   "Módulo de Expansión (16 Canales)",
        "Marque se a placa de expansão (8 canais extras) está conectada.":
            "Marque si la placa de expansión (8 canales extra) está conectada.",
        "16 canais marcados como EEG. Filtro 0.5-70 Hz + notch.":
            "16 canales como EEG. Filtro 0.5-70 Hz + notch.",
        "16 canais marcados como EMG. Filtro 20-Nyquist + notch.":
            "16 canales como EMG. Filtro 20-Nyquist + notch.",
        "16 canais marcados como ECG. Filtro 0.5-100 Hz + notch.":
            "16 canales como ECG. Filtro 0.5-100 Hz + notch.",
        "Mistura: 1-8 EEG / 9-12 EMG / 13-14 ECG / 15-16 EoG.":
            "Mezcla: 1-8 EEG / 9-12 EMG / 13-14 ECG / 15-16 EoG.",
        "Porta:":                           "Puerto:",
        "Modo:":                            "Modo:",
        "Atualizar lista de portas COM":    "Actualizar lista de puertos COM",
        "Setup pronto:":                    "Listo:",
        "Abrir o aplicativo direto na aba Configurações":
            "Abrir la aplicación directamente en la pestaña Ajustes",
        "Modo de Aquisição — Visibilidade de Abas":
            "Modo de Adquisición — Visibilidad de Pestañas",
        "Esconde abas que não fazem sentido para o modo escolhido (ex.: Topografia/ERP são EEG-only).":
            "Oculta pestañas que no aplican al modo elegido (ej.: Topografía/ERP son solo EEG).",
        # ===== GroupBoxes comuns =====
        "Tema (paleta de cores)":           "Tema (paleta de colores)",
        "Idioma / Language / Idioma":       "Idioma / Language",
        "Selecione:":                       "Seleccionar:",
        "Editor de Tema Personalizado":     "Editor de Tema Personalizado",
        "Filtro Notch (rejeição de banda)": "Filtro Notch (rechazo de banda)",
        "Filtro Bandpass (Butterworth, ordem 4)":
            "Filtro Pasabanda (Butterworth, orden 4)",
        "Ativado":                          "Activado",
        "Frequência (Hz):":                 "Frecuencia (Hz):",
        "Corte inferior (Hz):":             "Corte inferior (Hz):",
        "Corte superior (Hz):":             "Corte superior (Hz):",
        "Presets:":                         "Presets:",
        "Canais — Ativação e Tipo de Sinal (multimodal: EEG / EMG / ECG / EoG)":
            "Canales — Activación y Tipo de Señal (multimodal: EEG / EMG / ECG / EoG)",
        "Sessão e Arquivos":                "Sesión y Archivos",
        "Caminhos e Auditoria":             "Rutas y Auditoría",
        "Caminhos do Sistema (editáveis)":  "Rutas del Sistema (editables)",
        "Log de Auditoria (events.jsonl)":  "Registro de Auditoría (events.jsonl)",
        "Pasta de Salvamento das Sessões":  "Carpeta para Guardar Sesiones",
        "Exportar Sessão (escolher .csv ou pasta de sessão)":
            "Exportar Sesión (elegir .csv o carpeta)",
        "Aplicar agora":                    "Aplicar ahora",
        "Salvar como novo tema":            "Guardar como nuevo tema",
        "Deletar tema":                     "Eliminar tema",
        # ===== EMG / ECG / EoG / Focus / Joystick =====
        "Configuração do Envelope EMG":     "Configuración de la Envolvente EMG",
        "Método:":                          "Método:",
        "Janela (ms):":                     "Ventana (ms):",
        "Threshold global:":                "Umbral global:",
        "Envelope EMG — últimos 10 s":      "Envolvente EMG — últimos 10 s",
        "0 canais EMG ativos":              "0 canales EMG activos",
        "Canal ECG:":                       "Canal ECG:",
        "Sinal ECG (filtrado 5-15 Hz)":     "Señal ECG (filtrada 5-15 Hz)",
        "MWA (integral) + threshold":       "MWA (integral) + umbral",
        "Tacograma — intervalos RR (ms)":   "Tacograma — intervalos RR (ms)",
        "Poincaré — RR(n) × RR(n+1)":       "Poincaré — RR(n) × RR(n+1)",
        "Canal HEoG (horizontal):":         "Canal HEoG (horizontal):",
        "Canal VEoG (vertical):":           "Canal VEoG (vertical):",
        "Threshold (µV):":                  "Umbral (µV):",
        "Centro":                           "Centro",
        "Cima":                             "Arriba",
        "Baixo":                            "Abajo",
        "Esquerda":                         "Izquierda",
        "Direita":                          "Derecha",
        "piscadas detectadas":              "parpadeos detectados",
        "piscadas/min":                     "parpadeos/min",
        "Canal EEG:":                       "Canal EEG:",
        "Freq SSVEP alvo (Hz):":            "Frec SSVEP objetivo (Hz):",
        "Definir baseline (5s)":            "Definir línea base (5s)",
        "Estado":                           "Estado",
        "FOCADO":                           "ENFOCADO",
        "RELAXADO":                         "RELAJADO",
        "NORMAL":                           "NORMAL",
        "(sem baseline)":                   "(sin línea base)",
        "Mapeamento dos 4 canais EMG → eixos":
            "Mapeo de los 4 canales EMG → ejes",
        "Direção":                          "Dirección",
        "Canal EMG":                        "Canal EMG",
        "Envelope atual":                   "Envolvente actual",
        "Max calibrado":                    "Máx calibrado",
        "Calibrar (3s contração)":          "Calibrar (3s contracción)",
        "Calibrar":                         "Calibrar",
        "Dead zone:":                       "Zona muerta:",
        "Smoothing (frames):":              "Suavizado (frames):",
        "Sem canal":                        "Sin canal",
        "Selecione um canal EMG primeiro.": "Seleccione un canal EMG primero.",
        # ===== Conexão / Hardware =====
        "Modo de Aquisição":                "Modo de Adquisición",
        "Hardware (porta COM real)":        "Hardware (puerto COM real)",
        "Simulação (sinal sintético)":      "Simulación (señal sintética)",
        "Playback (replay de CSV)":         "Reproducción (CSV)",
        "Arquivo de playback:":             "Archivo de reproducción:",
        # ===== Voluntário =====
        "Novo voluntário":                  "Nuevo voluntario",
        "Editar voluntário":                "Editar voluntario",
        "Atualizar lista":                  "Actualizar lista",
        "Ativo: (nenhum)":                  "Activo: (ninguno)",
        "Histórico de sessões do voluntário selecionado":
            "Historial de sesiones del voluntario seleccionado",
        # ===== Banner Bio =====
        "Modalidades bioelétricas — placa multimodal Bionica Lab. Configure o tipo de cada canal em <b>Filtros e Canais → Tipo de sinal</b>.":
            "Modalidades bioeléctricas — placa multimodal Bionica Lab. Configura el tipo de cada canal en <b>Filtros y Canales → Tipo de señal</b>.",
        # ===== Status / Logs =====
        "Caminho copiado":                  "Ruta copiada",
        # ===== Settings =====
        "Tema:":   "Tema:",
        "Idioma:": "Idioma:",
        "Reinicie o app após trocar de idioma para aplicar em todas as telas.":
            "Reinicie la aplicación tras cambiar de idioma para aplicar en todas las pantallas.",
        # ===== Tipos de sinal =====
        "EEG": "EEG",
        "EMG": "EMG",
        "ECG": "ECG",
        "EoG": "EoG",
        "off": "off",
    }

    _maps = {"pt": {}, "en": _en, "es": _es}
    current = "pt"

    @classmethod
    def set_language(cls, lang):
        if lang in cls.LANGUAGES:
            cls.current = lang

    @classmethod
    def tr(cls, s):
        if cls.current == "pt":
            return s
        return cls._maps.get(cls.current, {}).get(s, s)


def tr(s):
    """Atalho global para I18N.tr(s)."""
    return I18N.tr(s)


# ============================================================
# AppConfig — persistencia (theme, mapping, session template)
# ============================================================
class AppConfig:
    def __init__(self, path=CONFIG_PATH):
        self.path = path
        self.theme = "Claro Clinico"
        self.language = "pt"  # pt / en / es
        # Aceite do Termo de Uso (assistente de primeiro uso)
        self.terms_accepted    = False
        self.terms_version     = ""
        self.terms_accepted_at = ""    # ISO-8601 local
        self.first_run_done    = False
        self.channel_mapping = list(DEFAULT_MAPPING)
        # Tipo de sinal por canal — Bionica Lab board é multimodal.
        # CH 1-8: EEG por padrão; CH 9-16: EMG por padrão (alvo músculo
        # após implementação da expansão multimodal).
        # MAX_CHANNELS = 16 — definido no escopo módulo
        # Default: CH 1-8 EEG; CH 9-16 EMG; CH 17+ EEG (placeholder até o
        # usuário configurar). Listas com tamanho MAX_CHANNELS para evitar
        # IndexError em qualquer modo (8 a 64 canais).
        _MC = MAX_CHANNELS if "MAX_CHANNELS" in globals() else 16
        self.channel_signal_types = (
            (["EEG"] * 8 + ["EMG"] * 8 + ["EEG"] * max(0, _MC - 16))
        )[: _MC]
        # Parâmetros EMG (envelope)
        self.emg_envelope_method = "RMS"   # RMS / |x|+LP / MAV
        self.emg_envelope_window_ms = 100   # janela de envelope (ms)
        self.emg_threshold_uV = [50.0] * _MC  # threshold por canal
        # Músculo mapeado por canal (chave de COMMON_MUSCLES)
        self.emg_channel_muscle = ["(não definido)"] * _MC
        # MVC (Maximum Voluntary Contraction) em µV por canal
        # Usado para normalizar envelope como % MVC
        self.emg_channel_mvc_uV = [0.0] * _MC
        self.session_template = "{subject}_{date}_{time}"
        self.subject = "voluntario"
        self.snapshot_interval = 30  # segundos; 0 = desativado
        self.custom_themes = {}      # {name: {color_key: hex, ...}}
        # Default: pasta `sessions/` ao lado do .py — o usuário pode trocar nas Configurações
        self.save_directory = DEFAULT_SAVE_DIRECTORY
        # Limites de impedância (kOhm) — personalizáveis pelo usuário na aba Calibração
        self.imp_good_max       = 10.0    # <= good = verde
        self.imp_acceptable_max = 50.0    # <= acceptable = amarelo; > = vermelho
        # Layout customizavel — kind+canal por slot e tamanhos do splitter
        self.layout_slots_cfg = [
            {"kind": "ts1",   "channel": 0},
            {"kind": "fft",   "channel": 0},
            {"kind": "head",  "channel": 0},
            {"kind": "bands", "channel": 0},
        ]
        self.layout_split_h_sizes = [600, 600]   # esquerda x direita
        self.layout_split_left    = [300, 300]   # cima x baixo (coluna esquerda)
        self.layout_split_right   = [300, 300]   # cima x baixo (coluna direita)
        self.load()
        # Registra temas customizados em THEMES
        for name, palette in self.custom_themes.items():
            THEMES[name] = palette
        # Garante que o diretório de salvamento existe
        try:
            os.makedirs(self.save_directory, exist_ok=True)
        except Exception:
            self.save_directory = DEFAULT_SAVE_DIRECTORY
            os.makedirs(self.save_directory, exist_ok=True)

    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.theme = d.get("theme", self.theme)
            lang = d.get("language", self.language)
            if isinstance(lang, str) and lang in ("pt", "en", "es"):
                self.language = lang
            # Aceite do Termo de Uso
            self.terms_accepted = bool(d.get("terms_accepted", self.terms_accepted))
            tv = d.get("terms_version", self.terms_version)
            if isinstance(tv, str): self.terms_version = tv
            ta = d.get("terms_accepted_at", self.terms_accepted_at)
            if isinstance(ta, str): self.terms_accepted_at = ta
            self.first_run_done = bool(d.get("first_run_done", self.first_run_done))
            mapping = d.get("channel_mapping", self.channel_mapping)
            if isinstance(mapping, list) and len(mapping) == len(DEFAULT_MAPPING):
                self.channel_mapping = mapping
            # Tipos de sinal por canal
            sigtypes = d.get("channel_signal_types", self.channel_signal_types)
            if isinstance(sigtypes, list):
                valid = [t if t in SIGNAL_TYPES else "EEG" for t in sigtypes]
                # Ajusta tamanho para MAX_CHANNELS
                if len(valid) < MAX_CHANNELS:
                    valid += ["EEG"] * (MAX_CHANNELS - len(valid))
                self.channel_signal_types = valid[: MAX_CHANNELS]
            # Parâmetros EMG
            em = d.get("emg_envelope_method")
            if isinstance(em, str) and em in ("RMS", "|x|+LP", "MAV"):
                self.emg_envelope_method = em
            ew = d.get("emg_envelope_window_ms")
            if isinstance(ew, (int, float)) and 10 <= ew <= 1000:
                self.emg_envelope_window_ms = float(ew)
            eth = d.get("emg_threshold_uV")
            if isinstance(eth, list) and len(eth) >= MAX_CHANNELS:
                try:
                    self.emg_threshold_uV = [float(x) for x in eth[: MAX_CHANNELS]]
                except Exception:
                    pass
            mus = d.get("emg_channel_muscle")
            if isinstance(mus, list):
                if len(mus) < MAX_CHANNELS:
                    mus += ["(não definido)"] * (MAX_CHANNELS - len(mus))
                self.emg_channel_muscle = [
                    m if m in COMMON_MUSCLES else "(não definido)"
                    for m in mus[: MAX_CHANNELS]
                ]
            mvc = d.get("emg_channel_mvc_uV")
            if isinstance(mvc, list) and len(mvc) >= MAX_CHANNELS:
                try:
                    self.emg_channel_mvc_uV = [float(x) for x in mvc[: MAX_CHANNELS]]
                except Exception:
                    pass
            self.session_template = d.get("session_template", self.session_template)
            self.subject = d.get("subject", self.subject)
            self.snapshot_interval = int(d.get("snapshot_interval", self.snapshot_interval))
            ct = d.get("custom_themes", {})
            if isinstance(ct, dict):
                self.custom_themes = ct
            sd = d.get("save_directory", "")
            if isinstance(sd, str) and sd.strip():
                self.save_directory = sd
            self.imp_good_max       = float(d.get("imp_good_max",       self.imp_good_max))
            self.imp_acceptable_max = float(d.get("imp_acceptable_max", self.imp_acceptable_max))
            slots = d.get("layout_slots_cfg")
            if (isinstance(slots, list) and len(slots) == 4
                    and all(isinstance(s, dict) for s in slots)):
                self.layout_slots_cfg = slots
            for key in ("layout_split_h_sizes", "layout_split_left", "layout_split_right"):
                val = d.get(key)
                if isinstance(val, list) and len(val) == 2:
                    setattr(self, key, val)
        except Exception as exc:
            print(f"[AppConfig] aviso: não foi possivel ler {self.path}: {exc}")

    def save(self):
        # Serializa ANTES de tocar o arquivo final: um erro de serializacao nao
        # destroi a config existente.
        try:
            data = json.dumps({
                    "theme": self.theme,
                    "language": self.language,
                    "terms_accepted":    self.terms_accepted,
                    "terms_version":     self.terms_version,
                    "terms_accepted_at": self.terms_accepted_at,
                    "first_run_done":    self.first_run_done,
                    "channel_mapping": self.channel_mapping,
                    "channel_signal_types": self.channel_signal_types,
                    "emg_envelope_method":   self.emg_envelope_method,
                    "emg_envelope_window_ms": self.emg_envelope_window_ms,
                    "emg_threshold_uV":      self.emg_threshold_uV,
                    "emg_channel_muscle":    self.emg_channel_muscle,
                    "emg_channel_mvc_uV":    self.emg_channel_mvc_uV,
                    "session_template": self.session_template,
                    "subject": self.subject,
                    "snapshot_interval": self.snapshot_interval,
                    "custom_themes": self.custom_themes,
                    "save_directory": self.save_directory,
                    "imp_good_max":       self.imp_good_max,
                    "imp_acceptable_max": self.imp_acceptable_max,
                    "layout_slots_cfg":     self.layout_slots_cfg,
                    "layout_split_h_sizes": self.layout_split_h_sizes,
                    "layout_split_left":    self.layout_split_left,
                    "layout_split_right":   self.layout_split_right,
                }, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[AppConfig] falha serializando config: {exc}")
            return
        # Escrita ATOMICA: grava em .tmp, fsync, backup .bak, e os.replace().
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(data); f.flush()
                try: os.fsync(f.fileno())
                except Exception: pass
            if os.path.exists(self.path):
                try:
                    import shutil
                    shutil.copyfile(self.path, self.path + ".bak")
                except Exception: pass
            os.replace(tmp, self.path)   # troca atomica (mesmo volume)
        except Exception as exc:
            print(f"[AppConfig] falha salvando {self.path}: {exc}")
            try: os.remove(tmp)
            except Exception: pass


# ============================================================
# VolunteerRegistry — cadastro de voluntários + ficha demográfica
# (portado/adaptado do "Data acquisition system.py" — VolunteerManager.
#  Versão limpa, sem dependência de pygame.)
#
# Estrutura:
#   <save_directory>/volunteers/
#     V01_NomeSobrenome/
#       profile.json   → ficha demográfica do voluntário
#       progress.json  → histórico de sessões executadas
#       V01_NomeSobrenome_<template>_<timestamp>/
#          data.csv, events.csv, session.log.txt, summary.json, snapshots/
# ============================================================
VOLUNTEER_PROFILE_FIELDS = [
    # (chave, rótulo, tipo, opções)
    ("nome",           "Nome completo",        "text",  None),
    ("idade",          "Idade (anos)",         "int",   None),
    ("sexo",           "Sexo",                 "enum",  ["M", "F", "Outro", "Prefiro não dizer"]),
    ("peso",           "Peso (kg)",            "float", None),
    ("altura",         "Altura (cm)",          "int",   None),
    ("profissao",      "Profissão",            "text",  None),
    ("mao_dominante",  "Mão dominante",        "enum",  ["Destra", "Canhota", "Ambidestra"]),
    ("escolaridade",   "Escolaridade",         "enum",  ["Fundamental", "Médio",
                                                          "Superior", "Pós-graduação"]),
    ("qualidade_sono", "Qualidade do sono",    "enum",  ["Excelente", "Boa",
                                                          "Regular", "Ruim", "Péssima"]),
    ("medicacao",      "Medicação em uso",     "text",  None),
    ("condicao",       "Condição clínica",     "text",  None),
    ("observacoes",    "Observações",          "text",  None),
]


class VolunteerRegistry:
    """Gerencia voluntários e suas fichas. Pasta `volunteers/` fica dentro
    do diretório de salvamento atual (configurável)."""

    def __init__(self, base_dir):
        self.set_base_dir(base_dir)
        self._current = None

    def set_base_dir(self, base_dir):
        self.base_dir = base_dir
        self.volunteers_dir = os.path.join(base_dir, "volunteers")
        try:
            os.makedirs(self.volunteers_dir, exist_ok=True)
        except Exception:
            pass

    @staticmethod
    def _safe_dirname(vid, name):
        clean = "".join(c if c.isalnum() else "_" for c in (name or "").strip())
        clean = "_".join(filter(None, clean.split("_")))[:30]
        return f"{vid}_{clean}" if clean else vid

    def list_volunteers(self):
        result = []
        if not os.path.isdir(self.volunteers_dir):
            return result
        for entry in sorted(os.listdir(self.volunteers_dir)):
            full = os.path.join(self.volunteers_dir, entry)
            prof_path = os.path.join(full, "profile.json")
            if os.path.isdir(full) and os.path.isfile(prof_path):
                try:
                    with open(prof_path, "r", encoding="utf-8") as f:
                        prof = json.load(f)
                    prof["_dirname"] = entry
                    result.append(prof)
                except Exception as exc:
                    print(f"[Volunteers] falha lendo {prof_path}: {exc}")
        return result

    def next_vid(self):
        nums = []
        for p in self.list_volunteers():
            vid = p.get("vid", "")
            if vid.startswith("V") and vid[1:].isdigit():
                nums.append(int(vid[1:]))
        n = (max(nums) + 1) if nums else 1
        return f"V{n:02d}"

    def create_volunteer(self, profile):
        vid = (profile.get("vid") or "").strip()
        name = (profile.get("nome") or "").strip()
        if not vid:
            raise ValueError("VID é obrigatório")
        if not name:
            raise ValueError("Nome é obrigatório")
        dirname = self._safe_dirname(vid, name)
        full = os.path.join(self.volunteers_dir, dirname)
        if os.path.isdir(full):
            raise ValueError(f"Voluntário já existe: {dirname}")
        os.makedirs(full)
        profile["_dirname"]   = dirname
        profile["created_at"] = datetime.now().isoformat(timespec="seconds")
        with open(os.path.join(full, "profile.json"), "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        with open(os.path.join(full, "progress.json"), "w", encoding="utf-8") as f:
            json.dump({"executions": [], "updated_at":
                       datetime.now().isoformat(timespec="seconds")},
                      f, indent=2, ensure_ascii=False)
        self._current = profile
        return profile

    def select_volunteer(self, dirname):
        full = os.path.join(self.volunteers_dir, dirname)
        prof_path = os.path.join(full, "profile.json")
        if not os.path.isfile(prof_path):
            raise FileNotFoundError(prof_path)
        with open(prof_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
        profile["_dirname"] = dirname
        self._current = profile
        return profile

    def clear_current(self):
        self._current = None

    def delete_volunteer(self, dirname):
        """Apaga a pasta completa do voluntário (incluindo sessões e ficha).
        Operação destrutiva — use com confirmação."""
        import shutil
        full = os.path.join(self.volunteers_dir, dirname)
        if not os.path.isdir(full):
            raise FileNotFoundError(dirname)
        shutil.rmtree(full)
        # Se era o ativo, limpa
        if self._current and self._current.get("_dirname") == dirname:
            self._current = None

    def current(self):
        return self._current

    def current_dir(self):
        if not self._current:
            return None
        return os.path.join(self.volunteers_dir, self._current["_dirname"])

    def get_progress(self, profile=None):
        profile = profile or self._current
        if not profile:
            return {"executions": []}
        path = os.path.join(self.volunteers_dir, profile["_dirname"],
                            "progress.json")
        if not os.path.isfile(path):
            return {"executions": []}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"executions": []}

    def add_execution(self, session_name, n_samples, n_markers, profile=None):
        profile = profile or self._current
        if not profile:
            return
        path = os.path.join(self.volunteers_dir, profile["_dirname"],
                            "progress.json")
        prog = self.get_progress(profile)
        prog.setdefault("executions", []).append({
            "session":   session_name,
            "at":        datetime.now().isoformat(timespec="seconds"),
            "samples":   int(n_samples),
            "markers":   int(n_markers),
        })
        prog["updated_at"] = datetime.now().isoformat(timespec="seconds")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(prog, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"[Volunteers] falha salvando progress: {exc}")


# ============================================================
# EventsLogger — events.csv mapeando marcadores → linhas do data.csv
# (inspirado no SessionLogger do "Data acquisition system.py").
# Permite análise futura por épocas: cada marcador guarda o índice
# de amostra (= linha de dados no data.csv) e o timestamp.
# ============================================================
class EventsLogger:
    def __init__(self):
        self._fp = None
        self._writer = None
        self._idx = 0
        self.path = None

    def start(self, path):
        self.stop()
        self.path = path
        try:
            self._fp = open(path, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._fp)
            self._writer.writerow([
                "event_idx", "label", "sample_index", "csv_data_line",
                "t_session_s", "timestamp_iso",
            ])
            self._fp.flush()
            self._idx = 0
            return True
        except Exception as exc:
            print(f"[Events] falha abrindo {path}: {exc}")
            self._fp = self._writer = None
            return False

    def log(self, label, sample_index, t_session_s):
        if self._writer is None:
            return
        try:
            self._idx += 1
            # csv_data_line: linha no data.csv (1 = primeira amostra; +1 por header)
            self._writer.writerow([
                self._idx, str(label), int(sample_index),
                int(sample_index) + 1,
                f"{t_session_s:.4f}",
                datetime.now().isoformat(timespec="milliseconds"),
            ])
            self._fp.flush()
        except Exception as exc:
            print(f"[Events] falha escrevendo: {exc}")

    def stop(self):
        if self._fp is not None:
            try: self._fp.close()
            except Exception: pass
        self._fp = self._writer = None


# ============================================================
# Parâmetros do sinal — suporte a expansão multi-step (8/16/24/32/40/48/56/64)
# ============================================================
BASE_CHANNELS  = 8                       # Cyton sozinho (placa base)
MAX_CHANNELS   = 64                      # capacidade máxima (placa base + 7 módulos de 8 canais)
CYTON_MAX_CHANNELS = 16                  # limite do protocolo OpenBCI Cyton+Daisy (legacy)
EXPANSION_STEPS = (8, 16, 24, 32, 40, 48, 56, 64)  # passos selecionáveis na aba Conexão
SAMPLE_RATE    = 250                     # Hz (250 sem expansão, 125 efetivo com Daisy; placa
                                         # customizada pode atingir taxas maiores)
BUFFER_SECONDS = 10
BUFFER_SIZE    = SAMPLE_RATE * BUFFER_SECONDS
SPEC_FMAX      = 60
SPEC_FRAMES    = 150
ACCEL_BUFFER_SIZE = SAMPLE_RATE * BUFFER_SECONDS

EEG_BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta":  (13.0, 30.0),
    "Gamma": (30.0, 50.0),
}

# ============================================================
# Tipos de sinal por canal — placa multimodal (EEG / EMG / ECG / EoG)
# ============================================================
# A placa em desenvolvimento (Bionica Lab) é multimodal: cada canal pode ser
# configurado para um tipo diferente de bioelétrico. Os parâmetros de filtro
# e o pré-processamento mudam conforme o tipo.
SIGNAL_TYPES = ("EEG", "EMG", "ECG", "EoG", "off")

# Presets de filtro recomendados por tipo de sinal.
# - EEG: 0.5-70 Hz (clínico); 1-45 Hz para ICA limpa
# - EMG: 20-450 Hz (limitado a Nyquist=125Hz a 250Hz); high-pass 20Hz remove deslocamento DC
# - ECG: 0.5-100 Hz; bandpass clínico (Einthoven)
# - EoG: 0.05-30 Hz; muito baixo HP para capturar movimento DC dos olhos
SIGNAL_FILTER_PRESETS = {
    "EEG": {"hp": 0.5, "lp": 70.0, "notch": True,  "label": "EEG (0.5-70 Hz)"},
    "EMG": {"hp": 20.0, "lp": min(450.0, SAMPLE_RATE/2 - 1), "notch": True,
            "label": "EMG (20 Hz - Nyquist)"},
    "ECG": {"hp": 0.5, "lp": 100.0, "notch": True,  "label": "ECG (0.5-100 Hz)"},
    "EoG": {"hp": 0.05, "lp": 30.0, "notch": True,  "label": "EoG (0.05-30 Hz)"},
    "off": {"hp": 0.5, "lp": 70.0, "notch": True,  "label": "Desativado"},
}

# Cor associada a cada tipo (usado em LEDs e na tabela de filtros)
SIGNAL_TYPE_COLORS = {
    "EEG": "#a3ff66",  # verde — sinal cerebral
    "EMG": "#ffaa55",  # laranja — sinal muscular
    "ECG": "#ff6677",  # vermelho — sinal cardíaco
    "EoG": "#66ddff",  # ciano — movimento ocular
    "off": "#666666",  # cinza — desativado
}

# ============================================================
# Mapeamento muscular para EMG — músculos comuns + ação/antagonista
# ============================================================
# Estrutura: nome -> {action: tipo de movimento, antagonist: músculo oposto,
#                     region: grupo anatômico}
# Usado para:
#   - Interpretar contração como flexão/extensão
#   - Detectar co-contração de antagonistas
#   - Sugerir agrupamentos sensoriais nos plots
COMMON_MUSCLES = {
    "(não definido)":            {"action": "",                  "antagonist": "",                       "region": ""},
    "Bíceps Braquial":           {"action": "Flexão cotovelo",   "antagonist": "Tríceps Braquial",       "region": "Braço"},
    "Tríceps Braquial":          {"action": "Extensão cotovelo", "antagonist": "Bíceps Braquial",        "region": "Braço"},
    "Deltoide Anterior":         {"action": "Flexão ombro",      "antagonist": "Deltoide Posterior",     "region": "Ombro"},
    "Deltoide Medio":            {"action": "Abdução ombro",     "antagonist": "",                       "region": "Ombro"},
    "Deltoide Posterior":        {"action": "Extensão ombro",    "antagonist": "Deltoide Anterior",      "region": "Ombro"},
    "Trapézio":                  {"action": "Elevação escápula", "antagonist": "",                       "region": "Costas"},
    "Latíssimo do Dorso":        {"action": "Adução ombro",      "antagonist": "Deltoide Medio",         "region": "Costas"},
    "Peitoral Maior":            {"action": "Adução horizontal", "antagonist": "Deltoide Posterior",     "region": "Tórax"},
    "Flexor Carpi Radialis":     {"action": "Flexão punho",      "antagonist": "Extensor Carpi Radialis","region": "Antebraço"},
    "Extensor Carpi Radialis":   {"action": "Extensão punho",    "antagonist": "Flexor Carpi Radialis",  "region": "Antebraço"},
    "Quadríceps (Reto Femoral)": {"action": "Extensão joelho",   "antagonist": "Isquiotibiais",          "region": "Coxa"},
    "Vasto Lateral":             {"action": "Extensão joelho",   "antagonist": "Isquiotibiais",          "region": "Coxa"},
    "Vasto Medial":              {"action": "Extensão joelho",   "antagonist": "Isquiotibiais",          "region": "Coxa"},
    "Isquiotibiais (Bíceps F.)": {"action": "Flexão joelho",     "antagonist": "Quadríceps (Reto Femoral)", "region": "Coxa"},
    "Glúteo Máximo":             {"action": "Extensão quadril",  "antagonist": "Iliopsoas",              "region": "Quadril"},
    "Iliopsoas":                 {"action": "Flexão quadril",    "antagonist": "Glúteo Máximo",          "region": "Quadril"},
    "Gastrocnêmio (medial)":     {"action": "Flexão plantar",    "antagonist": "Tibial Anterior",        "region": "Perna"},
    "Sóleo":                     {"action": "Flexão plantar",    "antagonist": "Tibial Anterior",        "region": "Perna"},
    "Tibial Anterior":           {"action": "Dorsiflexão",       "antagonist": "Gastrocnêmio (medial)",  "region": "Perna"},
    "Outro / Custom":            {"action": "",                  "antagonist": "",                       "region": ""},
}

# Aliases mais curtos para colunas estreitas (display only)
def _muscle_short(name):
    return {
        "Bíceps Braquial":            "Bíceps",
        "Tríceps Braquial":           "Tríceps",
        "Deltoide Anterior":          "Delt. Ant.",
        "Deltoide Medio":             "Delt. Med.",
        "Deltoide Posterior":         "Delt. Post.",
        "Latíssimo do Dorso":         "Latíssimo",
        "Peitoral Maior":             "Peitoral",
        "Flexor Carpi Radialis":      "Flexor punho",
        "Extensor Carpi Radialis":    "Ext. punho",
        "Quadríceps (Reto Femoral)":  "Quadríceps",
        "Vasto Lateral":              "Vasto Lat.",
        "Vasto Medial":               "Vasto Med.",
        "Isquiotibiais (Bíceps F.)":  "Isquio.",
        "Glúteo Máximo":              "Glúteo",
        "Iliopsoas":                  "Iliopsoas",
        "Gastrocnêmio (medial)":      "Gastroc.",
        "Sóleo":                      "Sóleo",
        "Tibial Anterior":            "Tib. Ant.",
    }.get(name, name)

# 16 posicoes 10-20 (-1..1, y+ = frente). CH1-8 = placa base; CH9-16 = módulo de expansão.
ELECTRODE_POSITIONS = [
    # ---- Base 8 (Cyton standard) ----
    ("Fp1", -0.25,  0.85), ("Fp2",  0.25,  0.85),
    ("C3",  -0.55,  0.00), ("C4",   0.55,  0.00),
    ("P7",  -0.65, -0.55), ("P8",   0.65, -0.55),
    ("O1",  -0.25, -0.85), ("O2",   0.25, -0.85),
    # ---- 1ª expansão (+8 = 16) ----
    ("F7",  -0.65,  0.55), ("F8",   0.65,  0.55),
    ("F3",  -0.35,  0.55), ("F4",   0.35,  0.55),
    ("T7",  -0.85,  0.00), ("T8",   0.85,  0.00),
    ("P3",  -0.35, -0.55), ("P4",   0.35, -0.55),
    # ---- 2ª expansão (+8 = 24) — linha medial ----
    ("Fz",   0.00,  0.55), ("Cz",   0.00,  0.00),
    ("Pz",   0.00, -0.55), ("Oz",   0.00, -0.85),
    ("FCz",  0.00,  0.28), ("CPz",  0.00, -0.28),
    ("FC3", -0.32,  0.28), ("FC4",  0.32,  0.28),
    # ---- 3ª expansão (+8 = 32) — centro-parietal ----
    ("CP3", -0.32, -0.28), ("CP4",  0.32, -0.28),
    ("FC1", -0.16,  0.28), ("FC2",  0.16,  0.28),
    ("CP1", -0.16, -0.28), ("CP2",  0.16, -0.28),
    ("C1",  -0.27,  0.00), ("C2",   0.27,  0.00),
    # ---- 4ª expansão (+8 = 40) — parietal e frontal ----
    ("P1",  -0.18, -0.55), ("P2",   0.18, -0.55),
    ("F1",  -0.18,  0.55), ("F2",   0.18,  0.55),
    ("AF3", -0.20,  0.72), ("AF4",  0.20,  0.72),
    ("PO3", -0.20, -0.72), ("PO4",  0.20, -0.72),
    # ---- 5ª expansão (+8 = 48) — linhas laterais ----
    ("F5",  -0.50,  0.55), ("F6",   0.50,  0.55),
    ("C5",  -0.72,  0.00), ("C6",   0.72,  0.00),
    ("P5",  -0.50, -0.55), ("P6",   0.50, -0.55),
    ("FT7", -0.80,  0.28), ("FT8",  0.80,  0.28),
    # ---- 6ª expansão (+8 = 56) — temporo-parietal e fronto-extremas ----
    ("TP7", -0.80, -0.28), ("TP8",  0.80, -0.28),
    ("FC5", -0.50,  0.28), ("FC6",  0.50,  0.28),
    ("CP5", -0.50, -0.28), ("CP6",  0.50, -0.28),
    ("AF7", -0.45,  0.72), ("AF8",  0.45,  0.72),
    # ---- 7ª expansão (+8 = 64) — occipital e antero-frontal ----
    ("PO7", -0.45, -0.72), ("PO8",  0.45, -0.72),
    ("POz",  0.00, -0.72), ("Fpz",  0.00,  0.85),
    ("AFz",  0.00,  0.72), ("Iz",   0.00, -0.92),
    ("T9",  -0.92,  0.00), ("T10",  0.92,  0.00),
]

# COLORS comeca com o tema default (Lime). Pode ser substituido em runtime
# via _apply_theme() — dict mutavel para que stylesheet/widgets que já leram
# as cores continuem funcionando (atualizamos as chaves in-place).
COLORS = dict(THEMES["Claro Clinico"])

CHANNEL_COLORS = [
    # CH1-8 (paleta original — placa base)
    "#a8ff00", "#00ffaa", "#00aaff", "#aa66ff",
    "#ff00aa", "#ffaa00", "#ff5500", "#55ff00",
    # CH9-16 (1ª expansão)
    "#88dd44", "#44ddaa", "#44aadd", "#bb88dd",
    "#dd44bb", "#ddaa44", "#dd6633", "#77dd44",
    # CH17-24 (2ª expansão — tons médios)
    "#66bb66", "#66ccaa", "#6699cc", "#9966cc",
    "#cc6699", "#cc9966", "#cc6633", "#99cc66",
    # CH25-32 (3ª expansão — tons claros)
    "#aaee88", "#aaeebb", "#88bbee", "#bb88ee",
    "#ee88bb", "#eebb88", "#ee9966", "#aaee66",
    # CH33-40 (4ª expansão — tons quentes)
    "#ffcc55", "#ff9966", "#ff6699", "#ff66cc",
    "#cc66ff", "#9966ff", "#6699ff", "#66ccff",
    # CH41-48 (5ª expansão — tons frios)
    "#66ffcc", "#99ff66", "#ccff66", "#ffff66",
    "#ffcc66", "#ff9966", "#ff6666", "#ff66aa",
    # CH49-56 (6ª expansão — pasteis)
    "#bbeebb", "#bbeecc", "#bbcccc", "#ccbbcc",
    "#ddbbcc", "#ddccbb", "#ddbbaa", "#ccddbb",
    # CH57-64 (7ª expansão — tons escuros)
    "#557755", "#557766", "#556677", "#665577",
    "#775566", "#776655", "#774422", "#447733",
]
# Garantia: sempre cobre MAX_CHANNELS
if len(CHANNEL_COLORS) < MAX_CHANNELS:
    CHANNEL_COLORS = CHANNEL_COLORS + ["#888888"] * (MAX_CHANNELS - len(CHANNEL_COLORS))
CHANNEL_COLORS = CHANNEL_COLORS[:MAX_CHANNELS]

# np.trapz foi removido no NumPy 2.4 (renomeado para np.trapezoid no 2.0).
# Usamos trapezoid quando existe (NumPy >= 2.0) e caímos para trapz no
# NumPy 1.x. getattr com default None evita avaliar um atributo ausente.
_TRAPEZOID = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


# ============================================================
# Versão e metadados do aplicativo
# ============================================================
APP_VERSION = "1.1.0"
APP_NAME       = "OpenBiônica"
APP_NAME_ASCII = "OpenBionica"   # forma sem acento p/ metadados ASCII (EDF/BIDS)
APP_EDITION    = "Edição Clínica"
APP_AUTHORS    = "OpenBiônica"
APP_YEAR    = 2026
TERMS_VERSION = "1.0"   # versao do Termo de Uso; bump => assistente reaparece p/ re-aceite
# Repositorio do codigo (mesmo fluxo do GitHub-pull / version.json) — usado para
# carimbar PROVENIENCIA nos arquivos gerados (reprodutibilidade/auditoria).
CODE_URL    = "https://github.com/rodrigooa43-create/OpenBionica"


# ============================================================
# Fontes da interface
# ============================================================
# Interface geral (botões, títulos, abas, labels descritivos) — sans serif moderna
FONT_UI         = "Inter"
FONT_UI_STACK   = "'Inter', 'Segoe UI', 'Roboto', Arial, sans-serif"
# Dados (tabelas, valores no plot, acelerômetro, log) — monoespaçada
FONT_DATA       = "JetBrains Mono"
FONT_DATA_STACK = "'JetBrains Mono', 'Cascadia Code', 'Consolas', 'Menlo', monospace"


# ============================================================
# Folha de estilo — função para reconstruir conforme tema
# ============================================================
def build_stylesheet(c=None):
    if c is None:
        c = COLORS
    return f"""
QWidget {{ background-color: {c['background']}; color: {c['text']};
           font-family: {FONT_UI_STACK}; font-size: 10pt; }}
QMainWindow {{ background-color: {c['background']}; }}
QTabWidget::pane {{ border: 1px solid {c['border']};
                    background-color: {c['surface']}; top: -1px; }}

/* ===== Estilo das abas =====
   Default = sub-abas (compactas). Mais abaixo, regras específicas para
   #mainTabs (top-level: grupos Configurar/Visualizar/Analisar/Sistema). */
QTabBar::tab {{ background-color: {c['surface']}; color: {c['text_dim']};
                padding: 7px 12px; border: 1px solid {c['border']};
                border-bottom: none; font-weight: 600; min-width: 90px; }}
QTabBar::tab:selected {{ background-color: {c['surface_alt']};
                         color: {c['accent']};
                         border-bottom: 2px solid {c['accent']}; }}
QTabBar::tab:hover:!selected {{ color: {c['accent_dim']};
                                background-color: {c['surface_alt']}; }}

/* === Top-level: grupos principais (maiores, mais destacados) === */
QTabWidget#mainTabs::pane {{ border: 1px solid {c['accent_dim']};
                              background-color: {c['surface']}; top: -1px;
                              border-top-left-radius: 0;
                              border-top-right-radius: 4px;
                              border-bottom-left-radius: 4px;
                              border-bottom-right-radius: 4px; }}
QTabWidget#mainTabs > QTabBar {{ qproperty-drawBase: 0; }}
QTabWidget#mainTabs > QTabBar::tab {{
    background-color: {c['surface']}; color: {c['text_dim']};
    padding: 11px 22px; border: 1px solid {c['border']};
    border-bottom: none; font-weight: bold; font-size: 11pt;
    min-width: 130px; letter-spacing: 1px;
    margin-right: 2px;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
}}
QTabWidget#mainTabs > QTabBar::tab:selected {{
    background-color: {c['accent_dim']};
    color: {c['background']};
    border: 1px solid {c['accent']};
    border-bottom: none;
}}
QTabWidget#mainTabs > QTabBar::tab:hover:!selected {{
    background-color: {c['surface_alt']};
    color: {c['accent']};
}}

/* === Sub-level: abas internas (menores, mais sutis) === */
QTabWidget#subTabs::pane {{ border: 1px solid {c['border']};
                             background-color: {c['surface']}; top: -1px; }}
QTabWidget#subTabs > QTabBar::tab {{
    background-color: transparent;
    color: {c['text_dim']};
    padding: 6px 14px;
    border: none;
    border-bottom: 2px solid transparent;
    font-weight: 600; font-size: 10pt;
    min-width: 110px;
    margin-right: 0px;
}}
QTabWidget#subTabs > QTabBar::tab:selected {{
    color: {c['accent']};
    border-bottom: 2px solid {c['accent']};
    background-color: transparent;
}}
QTabWidget#subTabs > QTabBar::tab:hover:!selected {{
    color: {c['accent_dim']};
    border-bottom: 2px solid {c['accent_dim']};
}}
QPushButton {{ background-color: {c['surface_alt']}; color: {c['accent']};
               border: 1px solid {c['accent_dim']}; padding: 6px 12px;
               font-weight: bold; border-radius: 4px; }}
QPushButton:hover {{ background-color: {c['accent_dim']}; color: {c['background']}; }}
QPushButton:pressed {{ background-color: {c['accent']}; color: {c['background']}; }}
QPushButton:disabled {{ background-color: {c['surface']}; color: {c['text_dim']};
                        border-color: {c['border']}; }}
QPushButton:checked {{ background-color: {c['accent_dim']}; color: {c['background']}; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background-color: {c['surface_alt']};
    color: {c['text']}; border: 1px solid {c['border']}; padding: 4px 8px;
    border-radius: 3px; selection-background-color: {c['accent_dim']}; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border-color: {c['accent']}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background-color: {c['surface_alt']};
    color: {c['text']}; selection-background-color: {c['accent_dim']};
    selection-color: {c['background']}; border: 1px solid {c['accent']}; }}
QGroupBox {{ border: 1px solid {c['border']}; border-radius: 4px;
             margin-top: 12px; padding: 14px 10px 10px 10px; font-weight: bold;
             color: {c['accent']}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; }}
QCheckBox {{ color: {c['text']}; spacing: 6px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {c['border']};
    background: {c['surface_alt']}; border-radius: 2px; }}
QCheckBox::indicator:checked {{ background: {c['accent']}; border-color: {c['accent']}; }}
QLabel {{ color: {c['text']}; background-color: transparent; }}
QTableWidget {{ background-color: {c['table_bg']}; color: {c['text']};
    gridline-color: {c['border']}; alternate-background-color: {c['table_alt']};
    border: 1px solid {c['border']};
    font-family: {FONT_DATA_STACK}; }}
QTableWidget::item {{ background-color: {c['table_bg']}; color: {c['text']};
    font-family: {FONT_DATA_STACK}; }}
QTableWidget::item:alternate {{ background-color: {c['table_alt']}; }}
QTableWidget::item:selected {{ background-color: {c['accent_dim']};
    color: {c['background']}; }}
QHeaderView::section {{ background-color: {c['surface']}; color: {c['accent']};
    padding: 6px; border: 1px solid {c['border']}; font-weight: bold;
    font-family: {FONT_UI_STACK}; }}
QTableCornerButton::section {{ background-color: {c['surface']};
    border: 1px solid {c['border']}; }}
QScrollBar:vertical {{ background-color: {c['surface']}; width: 12px; border: none; }}
QScrollBar::handle:vertical {{ background-color: {c['accent_dim']};
    border-radius: 6px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background-color: {c['accent']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QTextEdit {{ background-color: {c['surface_alt']}; color: {c['text']};
    border: 1px solid {c['border']}; font-family: {FONT_DATA_STACK}; }}
QSlider::groove:horizontal {{ border: 1px solid {c['border']}; height: 6px;
    background: {c['surface_alt']}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {c['accent']};
    border: 1px solid {c['accent_dim']}; width: 14px; margin: -5px 0; border-radius: 7px; }}
QProgressBar {{ border: 1px solid {c['border']}; border-radius: 3px;
    background: {c['surface_alt']}; text-align: center; color: {c['text']}; }}
QProgressBar::chunk {{ background: {c['accent']}; }}
QMenuBar {{ background-color: {c['surface']}; color: {c['text']};
    border-bottom: 1px solid {c['border']}; }}
QMenuBar::item {{ background: transparent; padding: 5px 11px; }}
QMenuBar::item:selected {{ background: {c['surface_alt']}; color: {c['accent']};
    border-radius: 4px; }}
QMenu {{ background-color: {c['surface']}; color: {c['text']};
    border: 1px solid {c['border']}; padding: 4px; }}
QMenu::item {{ padding: 6px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background-color: {c['accent_dim']}; color: {c['background']}; }}
QStatusBar {{ background-color: {c['surface']}; color: {c['text_dim']};
    border-top: 1px solid {c['border']}; }}
QStatusBar::item {{ border: none; }}
QToolTip {{ background-color: {c['surface_alt']}; color: {c['text']};
    border: 1px solid {c['border']}; padding: 4px 6px; }}
"""


# Mantida para compatibilidade com codigo legado (recriada apos cada troca de tema)
STYLESHEET = build_stylesheet(COLORS)


def _apply_theme_colors(theme_name):
    """Atualiza COLORS (in-place) para o tema escolhido. Retorna o dict."""
    palette = THEMES.get(theme_name, THEMES["Lime (verde-limao)"])
    # mutacao in-place para que referencias existentes acompanhem
    for k, v in palette.items():
        COLORS[k] = v
    # Fallback: temas antigos sem table_bg/table_alt — derivam de surface/surface_alt
    if "table_bg"  not in palette: COLORS["table_bg"]  = palette.get("surface", "#1a1a1a")
    if "table_alt" not in palette: COLORS["table_alt"] = palette.get("surface_alt", "#252525")
    return COLORS


# ============================================================
# FilterChain — notch + bandpass por canal (com num canais variável)
# ============================================================
class FilterChain:
    # Esquemas de re-referenciação suportados
    REREF_NONE       = "none"        # Sem re-referenciação
    REREF_CAR        = "car"         # Common Average Reference (média de todos os canais)
    REREF_MASTOID    = "mastoid"     # Média de canais de referência (M1+M2)/2
    REREF_LAPLACIAN  = "laplacian"   # Surface Laplacian (canal - média vizinhos)
    REREF_BIPOLAR    = "bipolar"     # Pares bipolares (ch i+1 - ch i)
    REREF_REST       = "rest"        # Aproximação REST (subtrai média de canais distantes)

    def __init__(self, sample_rate=SAMPLE_RATE, num_channels=MAX_CHANNELS):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.notch_enabled    = False
        self.notch_freq       = 60.0
        self.bandpass_enabled = False
        self.bp_low           = 1.0
        self.bp_high          = 50.0
        # ---- Re-referenciação ----
        self.reref_mode = self.REREF_NONE
        # Canais de referência (índices) para esquemas que precisam (Mastoid)
        self.reref_channels = []
        # Vizinhos por canal (para Laplacian) — pré-computado em set_reref_mode
        self._laplacian_neighbors = None
        self._sos_notch = None
        self._zi_notch  = None
        self._sos_bp    = None
        self._zi_bp     = None
        self.rebuild()

    def set_reref(self, mode, ref_channels=None):
        """Configura modo de re-referenciação. ref_channels: índices 0-based."""
        if mode in (self.REREF_NONE, self.REREF_CAR, self.REREF_MASTOID,
                    self.REREF_LAPLACIAN, self.REREF_BIPOLAR, self.REREF_REST):
            self.reref_mode = mode
        if ref_channels is not None:
            self.reref_channels = list(ref_channels)
        # Pre-compute neighbors for Laplacian (uses ELECTRODE_POSITIONS distances)
        if self.reref_mode == self.REREF_LAPLACIAN:
            self._compute_laplacian_neighbors()

    def _compute_laplacian_neighbors(self):
        """Para cada canal, identifica os 3-4 vizinhos mais próximos via posição 10-20."""
        try:
            positions = [(p[1], p[2]) for p in ELECTRODE_POSITIONS]
        except Exception:
            self._laplacian_neighbors = None; return
        n = min(self.num_channels, len(positions))
        neighbors = []
        for i in range(n):
            d = []
            for j in range(n):
                if j == i: continue
                dist = ((positions[i][0] - positions[j][0])**2 +
                        (positions[i][1] - positions[j][1])**2) ** 0.5
                d.append((dist, j))
            d.sort()
            # Pega 4 vizinhos mais próximos (Laplaciano local)
            neighbors.append([j for _, j in d[:4]])
        self._laplacian_neighbors = neighbors

    def apply_reref(self, sample):
        """Aplica re-referenciação a um vetor de amostras (in-place safe)."""
        n = len(sample)
        if n < 2 or self.reref_mode == self.REREF_NONE:
            return sample
        out = sample.copy()
        if self.reref_mode == self.REREF_CAR:
            ref = float(np.mean(out[:n]))
            out[:n] -= ref
        elif self.reref_mode == self.REREF_MASTOID:
            chans = [c for c in self.reref_channels if 0 <= c < n]
            if chans:
                ref = float(np.mean(out[chans]))
                out[:n] -= ref
        elif self.reref_mode == self.REREF_LAPLACIAN and self._laplacian_neighbors:
            tmp = out.copy()
            for i in range(min(n, len(self._laplacian_neighbors))):
                nb = [j for j in self._laplacian_neighbors[i] if j < n]
                if nb:
                    out[i] = tmp[i] - float(np.mean(tmp[nb]))
        elif self.reref_mode == self.REREF_BIPOLAR:
            # ch i' = ch i - ch i+1 (último mantido)
            tmp = out.copy()
            for i in range(n - 1):
                out[i] = tmp[i] - tmp[i + 1]
        elif self.reref_mode == self.REREF_REST:
            # Aproximação simplificada: subtrai média dos canais marcados como
            # 'distantes' (= todos exceto os 4 vizinhos)
            if self._laplacian_neighbors:
                tmp = out.copy()
                for i in range(min(n, len(self._laplacian_neighbors))):
                    far = [j for j in range(n)
                           if j != i and j not in self._laplacian_neighbors[i]]
                    if far:
                        out[i] = tmp[i] - float(np.mean(tmp[far]))
            else:
                ref = float(np.mean(out[:n]))
                out[:n] -= ref
        return out

    def rebuild(self):
        nyq = self.sample_rate / 2.0
        if self.notch_enabled and 0 < self.notch_freq < nyq:
            b, a = scipy_signal.iirnotch(self.notch_freq / nyq, Q=30.0)
            self._sos_notch = scipy_signal.tf2sos(b, a)
            zi = scipy_signal.sosfilt_zi(self._sos_notch)
            self._zi_notch = np.stack([zi.copy() for _ in range(self.num_channels)])
        else:
            self._sos_notch = self._zi_notch = None
        if self.bandpass_enabled and 0 < self.bp_low < self.bp_high < nyq:
            self._sos_bp = scipy_signal.butter(
                4, [self.bp_low / nyq, self.bp_high / nyq],
                btype="bandpass", output="sos",
            )
            zi = scipy_signal.sosfilt_zi(self._sos_bp)
            self._zi_bp = np.stack([zi.copy() for _ in range(self.num_channels)])
        else:
            self._sos_bp = self._zi_bp = None

    def reset_state(self):
        if self._zi_notch is not None:
            for ch in range(self.num_channels):
                self._zi_notch[ch] *= 0
        if self._zi_bp is not None:
            for ch in range(self.num_channels):
                self._zi_bp[ch] *= 0

    def apply_sample(self, sample):
        """Aplica re-ref + notch + bandpass aos canais ativos.

        Ordem: re-referenciação → notch → bandpass (padrão da literatura).
        """
        # 1) Re-referenciação (CAR / Laplacian / Mastoid / Bipolar / REST)
        if self.reref_mode != self.REREF_NONE:
            out = self.apply_reref(sample)
        else:
            out = sample.copy()
        n = len(out)
        if self._sos_notch is not None:
            for ch in range(min(n, self.num_channels)):
                y, self._zi_notch[ch] = scipy_signal.sosfilt(
                    self._sos_notch, [out[ch]], zi=self._zi_notch[ch]
                )
                out[ch] = y[0]
        if self._sos_bp is not None:
            for ch in range(min(n, self.num_channels)):
                y, self._zi_bp[ch] = scipy_signal.sosfilt(
                    self._sos_bp, [out[ch]], zi=self._zi_bp[ch]
                )
                out[ch] = y[0]
        return out


# ============================================================
# UDPSender
# ============================================================
class UDPSender:
    def __init__(self):
        self.host = "127.0.0.1"
        self.port = 12345
        self.enabled = False
        self._sock = None

    def start(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.enabled = True
            return True
        except Exception:
            self._sock = None; self.enabled = False; return False

    def stop(self):
        if self._sock:
            try: self._sock.close()
            except Exception: pass
        self._sock = None; self.enabled = False

    def send_sample(self, t, sample):
        if not self.enabled or self._sock is None: return
        try:
            payload = json.dumps({
                "t": round(t, 4),
                "n": len(sample),
                "v": [round(float(x), 3) for x in sample],
            }).encode("utf-8")
            self._sock.sendto(payload, (self.host, self.port))
        except Exception: pass

    def send_marker(self, t, label):
        if not self.enabled or self._sock is None: return
        try:
            payload = json.dumps({"t": round(t, 4), "marker": str(label)}).encode("utf-8")
            self._sock.sendto(payload, (self.host, self.port))
        except Exception: pass


# ============================================================
# LSLSender — Lab Streaming Layer (padrão de neurociência)
# ============================================================
class LSLSender:
    """Stream EEG via LSL (Lab Streaming Layer) — sub-ms sync com
    PsychoPy, OpenViBE, Unity, Matlab e centenas de outros softwares."""

    def __init__(self):
        self.enabled  = False
        self.outlet   = None
        self.marker_outlet = None
        self.stream_name   = "EEG_Data_Collector"
        self.source_id     = "eeg_collector_lime"

    @staticmethod
    def available():
        return HAS_LSL

    def start(self, num_channels=8, sample_rate=SAMPLE_RATE, ch_labels=None):
        if not HAS_LSL:
            return False
        try:
            import pylsl  # lazy: só carrega quando o usuário usa LSL
            info = pylsl.StreamInfo(
                name=self.stream_name,
                type="EEG",
                channel_count=num_channels,
                nominal_srate=sample_rate,
                channel_format=pylsl.cf_float32,
                source_id=self.source_id,
            )
            # Channel labels (eletrodos do mapeamento)
            chans = info.desc().append_child("channels")
            for i in range(num_channels):
                ch = chans.append_child("channel")
                lbl = (ch_labels[i] if ch_labels and i < len(ch_labels)
                       else f"CH{i+1}")
                ch.append_child_value("label", lbl)
                ch.append_child_value("unit", "microvolts")
                ch.append_child_value("type", "EEG")
            self.outlet = pylsl.StreamOutlet(info)

            # Outlet de markers separado (padrao LSL)
            mk_info = pylsl.StreamInfo(
                name=f"{self.stream_name}_markers", type="Markers",
                channel_count=1, nominal_srate=0,
                channel_format=pylsl.cf_string,
                source_id=f"{self.source_id}_markers",
            )
            self.marker_outlet = pylsl.StreamOutlet(mk_info)
            self.enabled = True
            return True
        except Exception as exc:
            print(f"[LSL] falha iniciando outlet: {exc}")
            self.outlet = self.marker_outlet = None
            self.enabled = False
            return False

    def stop(self):
        self.outlet = None
        self.marker_outlet = None
        self.enabled = False

    def send_sample(self, sample):
        if not self.enabled or self.outlet is None: return
        try:
            self.outlet.push_sample([float(x) for x in sample])
        except Exception: pass

    def send_marker(self, label):
        if not self.enabled or self.marker_outlet is None: return
        try:
            self.marker_outlet.push_sample([str(label)])
        except Exception: pass


# ============================================================
# AutoStats — FACILITADOR estatístico: escolhe o teste sozinho
# ============================================================
# Objetivo: o usuário (clínico/estudante) NÃO precisa saber qual teste usar.
# O fluxo padrão da literatura é automatizado: descritivas -> normalidade
# (Shapiro) -> paramétrico/não-paramétrico -> tamanho de efeito -> correção de
# múltiplas comparações (Holm) -> conclusão em linguagem simples.
def _stat_descr(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"),
                "median": float("nan"), "iqr": float("nan")}
    return {"n": int(a.size), "mean": float(np.mean(a)),
            "sd": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
            "median": float(np.median(a)),
            "iqr": float(np.subtract(*np.percentile(a, [75, 25])))}


def _cohen_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                 / (na + nb - 2))
    return float((np.mean(a) - np.mean(b)) / sp) if sp > 0 else float("nan")


def _effect_mag(d):
    d = abs(d)
    if not np.isfinite(d):
        return "—"
    if d < 0.2: return "desprezível"
    if d < 0.5: return "pequeno"
    if d < 0.8: return "médio"
    return "grande"


def auto_compare(groups, names=None, paired=False, alpha=0.05):
    """Escolhe e roda o teste estatístico adequado automaticamente.
    `groups`: lista de sequências numéricas (2 ou mais grupos)."""
    import scipy.stats as st
    arrs = [np.asarray([v for v in g if v is not None and np.isfinite(v)], float)
            for g in groups]
    names = names or [f"Grupo {i + 1}" for i in range(len(arrs))]
    res = {"names": names, "descr": [_stat_descr(g) for g in arrs], "alpha": alpha,
           "test": "—", "stat": float("nan"), "p": float("nan"),
           "effect_name": "—", "effect": float("nan"),
           "normal": None, "ok": False, "note": ""}
    if any(a.size < 2 for a in arrs):
        res["note"] = "amostra insuficiente (n<2 em algum grupo)"
        return res

    def _is_normal(a):
        if a.size < 3:
            return True
        try:
            return float(st.shapiro(a).pvalue) > 0.05
        except Exception:
            return True
    all_normal = all(_is_normal(a) for a in arrs)
    res["normal"] = bool(all_normal)
    try:
        if len(arrs) == 2:
            a, b = arrs
            if paired:
                n = min(len(a), len(b)); a, b = a[:n], b[:n]
                # No teste pareado o pressuposto e a normalidade das DIFERENCAS
                # d = a - b, nao de cada grupo isolado.
                diff = a - b
                paired_normal = _is_normal(diff)
                res["normal"] = bool(paired_normal)
                if paired_normal:
                    s, p = st.ttest_rel(a, b); res["test"] = "t de Student (pareado)"
                    sd = np.std(diff, ddof=1)
                    res["effect"] = float(np.mean(diff) / sd) if sd > 0 else float("nan")
                    res["effect_name"] = "Cohen dz"
                else:
                    s, p = st.wilcoxon(a, b); res["test"] = "Wilcoxon (pareado)"
                    res["effect_name"] = "r"
            else:
                if all_normal:
                    s, p = st.ttest_ind(a, b, equal_var=False)
                    res["test"] = "t de Welch (independente)"
                    res["effect"] = _cohen_d(a, b); res["effect_name"] = "Cohen d"
                else:
                    s, p = st.mannwhitneyu(a, b, alternative="two-sided")
                    res["test"] = "Mann-Whitney U"; res["effect_name"] = "r"
            res["stat"] = float(s); res["p"] = float(p)
        else:
            if all_normal:
                s, p = st.f_oneway(*arrs); res["test"] = "ANOVA (1 fator)"
            else:
                s, p = st.kruskal(*arrs); res["test"] = "Kruskal-Wallis"
            res["stat"] = float(s); res["p"] = float(p)
        res["ok"] = True
    except Exception as exc:
        res["note"] = f"falha no teste: {exc}"
    return res


def holm_correction(pvals):
    """Holm-Bonferroni: p-valores ajustados (mantém a ordem de entrada)."""
    p = np.asarray(pvals, float)
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    m = int(np.sum(np.isfinite(p)))
    adj = np.full_like(p, np.nan)
    running, rank = 0.0, 0
    for idx in order:
        if not np.isfinite(p[idx]):
            continue
        rank += 1
        running = max(running, (m - rank + 1) * float(p[idx]))
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def interpret_result(res):
    """Conclusão em linguagem simples (pt-BR) a partir de auto_compare()."""
    if not res.get("ok"):
        return f"Não foi possível concluir: {res.get('note') or 'dados insuficientes'}."
    p = res["p"]; nm = res["names"]; sig = p < res["alpha"]
    eff, en = res.get("effect"), res.get("effect_name")
    txt = (f"{res['test']}: " +
           ("DIFERENÇA significativa" if sig else "sem diferença significativa") +
           f" (p = {p:.4f}).")
    if len(nm) == 2 and sig:
        d = res["descr"]
        maior = nm[0] if d[0]["mean"] >= d[1]["mean"] else nm[1]
        txt += f" Em média, {maior} é maior."
    if en not in ("—", None) and eff is not None and np.isfinite(eff):
        txt += f" Efeito ({en}) = {eff:.2f} ({_effect_mag(eff)})."
    if any(d["n"] < 5 for d in res["descr"]):
        txt += " ATENÇÃO: amostra pequena (n<5) — interprete com cautela."
    return txt


# ============================================================
# GuidedStatsDialog — UI do facilitador estatístico
# ============================================================
class GuidedStatsDialog(QtWidgets.QDialog):
    """Compara grupos de sessões (ex.: Antes × Depois) por banda de potência,
    escolhe o teste sozinho, monta a tabela e explica — sem exigir conhecimento
    estatístico do usuário."""

    COLS = ["banda", "A", "B", "teste", "p", "p_holm", "efeito", "signif"]

    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self._files = {"A": [], "B": []}
        self._rows = []
        self.setWindowTitle("Estatística guiada — comparar grupos")
        self.resize(860, 580)
        v = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Selecione as sessões de cada grupo (ex.: <b>Antes</b> × <b>Depois</b>). "
            "Você pode juntar sessões de <b>vários pacientes</b> no mesmo grupo — "
            "ex.: o \"antes\" de todos num grupo e o \"depois\" de todos no outro. "
            "Cada <b>sessão conta como 1 amostra</b>.<br>"
            "<b>Regra:</b> cada grupo precisa de <b>≥ 2 sessões</b> para rodar o "
            "teste; <b>≥ 5</b> para um resultado confiável. O programa escolhe o "
            "teste, monta a tabela por banda (δ θ α β γ) e explica em linguagem "
            "simples. Aceita <b>.csv</b> e também <b>.edf/.bdf</b> (converte na hora).")
        intro.setWordWrap(True); intro.setTextFormat(QtCore.Qt.TextFormat.RichText)
        v.addWidget(intro)
        for key, default in (("A", "Antes"), ("B", "Depois")):
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(f"Grupo {key}:"))
            name = QtWidgets.QLineEdit(default); setattr(self, f"name_{key}", name)
            row.addWidget(name)
            cnt = QtWidgets.QLabel("0 sessões"); setattr(self, f"cnt_{key}", cnt)
            row.addWidget(cnt, 1)
            btn = QtWidgets.QPushButton("Selecionar sessões…")
            btn.clicked.connect(lambda _, k=key: self._pick(k))
            row.addWidget(btn)
            v.addLayout(row)
        self.paired_chk = QtWidgets.QCheckBox(
            "Amostras pareadas (os mesmos sujeitos nos dois grupos)")
        v.addWidget(self.paired_chk)
        run = QtWidgets.QPushButton("Comparar")
        run.clicked.connect(self._compute)
        v.addWidget(run)
        self.table = QtWidgets.QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(
            ["Banda", "Grupo A", "Grupo B", "Teste", "p", "p (Holm)",
             "Efeito", "Signif.?"])
        self.table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.table, 1)
        self.summary = QtWidgets.QTextEdit(); self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(140)
        v.addWidget(self.summary)
        save = QtWidgets.QPushButton("Salvar relatório (HTML + CSV)…")
        save.clicked.connect(self._save)
        v.addWidget(save)

    def _pick(self, key):
        start = getattr(self.main.config, "save_directory", "")
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, f"Sessões do Grupo {key}", start,
            "Sessões (*.csv *.edf *.bdf *.EDF *.BDF);;CSV (*.csv);;Todos (*)")
        if not files:
            return
        # Converte EDF/BDF -> CSV nativo (temporário) automaticamente.
        resolved = []
        for f in files:
            if os.path.splitext(f)[1].lower() in (".edf", ".bdf"):
                try:
                    import tempfile
                    tmp = os.path.join(
                        tempfile.gettempdir(),
                        f"edfimp_{os.path.splitext(os.path.basename(f))[0][:30]}.csv")
                    edf_to_native_csv(f, tmp)
                    resolved.append(tmp)
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(
                        self, "EDF", f"Não consegui ler {os.path.basename(f)}:\n{exc}")
            else:
                resolved.append(f)
        self._files[key] = resolved
        lbl = getattr(self, f"cnt_{key}")
        n = len(resolved)
        if n >= 5:
            lbl.setText(f"{n} sessões ✓ (bom)"); lbl.setStyleSheet("color:#1E7E34;")
        elif n >= 2:
            lbl.setText(f"{n} sessões ✓ (mínimo; ≥5 é melhor)")
            lbl.setStyleSheet("color:#B8860B;")
        else:
            lbl.setText(f"{n} sessão — precisa de ≥ 2")
            lbl.setStyleSheet("color:#C0392B;")

    def _band_powers(self, csv_path):
        d = self.main._load_session_csv(csv_path)
        if not d:
            return None
        eeg, sr = d["eeg"], d["sr"]
        out = {}
        for band, (lo, hi) in EEG_BANDS.items():
            vals = []
            for i in range(eeg.shape[0]):
                f, psd = scipy_signal.welch(eeg[i], fs=sr,
                                            nperseg=min(256, eeg.shape[1]))
                m = (f >= lo) & (f < hi)
                vals.append(float(_TRAPEZOID(psd[m], f[m])) if np.any(m) else 0.0)
            out[band] = float(np.mean(vals)) if vals else 0.0
        return out

    def _compute(self):
        A, B = self._files["A"], self._files["B"]
        if not A or not B:
            QtWidgets.QMessageBox.warning(
                self, "Grupos", "Selecione sessões nos dois grupos.")
            return None
        nameA = self.name_A.text().strip() or "A"
        nameB = self.name_B.text().strip() or "B"
        powA = [p for p in (self._band_powers(c) for c in A) if p]
        powB = [p for p in (self._band_powers(c) for c in B) if p]
        if not powA or not powB:
            QtWidgets.QMessageBox.critical(
                self, "Dados", "Não consegui ler potências de banda das sessões.")
            return None
        # Checagem de amostra ACIONÁVEL (o beta tester travava aqui sem entender).
        if len(powA) < 2 or len(powB) < 2:
            QtWidgets.QMessageBox.warning(
                self, "Amostra insuficiente",
                f"Para comparar, cada grupo precisa de pelo menos 2 sessões — "
                f"cada sessão conta como 1 amostra, e não dá para estimar a "
                f"variação com uma só.\n\n"
                f"Você tem agora:   {nameA} = {len(powA)}   |   {nameB} = {len(powB)}\n\n"
                f"Dica: você pode juntar sessões de VÁRIOS pacientes no mesmo grupo "
                f"(o \"antes\" de todos num grupo, o \"depois\" de todos no outro).\n"
                f"Recomendado: 5 ou mais por grupo para um resultado confiável.")
            return None
        bands = list(EEG_BANDS.keys())
        results, pvals = [], []
        for band in bands:
            ga = [p[band] for p in powA]; gb = [p[band] for p in powB]
            r = auto_compare([ga, gb], [nameA, nameB],
                             paired=self.paired_chk.isChecked())
            results.append((band, r)); pvals.append(r["p"])
        adj = holm_correction(pvals)
        self._rows = []
        self.table.setRowCount(len(results))
        for i, (band, r) in enumerate(results):
            da, db = r["descr"]
            eff = r.get("effect")
            eff_txt = (f"{r['effect_name']}={eff:.2f}"
                       if (eff is not None and np.isfinite(eff)
                           and r["effect_name"] not in ("—", None)) else "—")
            cells = {
                "banda": band,
                "A": f"{da['mean']:.3g}±{da['sd']:.2g} (n={da['n']})",
                "B": f"{db['mean']:.3g}±{db['sd']:.2g} (n={db['n']})",
                "teste": r["test"],
                "p": f"{r['p']:.4f}" if np.isfinite(r["p"]) else "—",
                "p_holm": f"{adj[i]:.4f}" if np.isfinite(adj[i]) else "—",
                "efeito": eff_txt,
                "signif": "SIM" if (np.isfinite(adj[i]) and adj[i] < 0.05) else "não",
            }
            for j, key in enumerate(self.COLS):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(str(cells[key])))
            self._rows.append(cells)
        self.table.setHorizontalHeaderLabels(
            ["Banda", nameA, nameB, "Teste", "p", "p (Holm)", "Efeito", "Signif.?"])
        self.table.resizeColumnsToContents()
        sig = [band for (band, _r), a in zip(results, adj)
               if np.isfinite(a) and a < 0.05]
        lines = [f"Comparação: {nameA} ({len(powA)} sessões) × "
                 f"{nameB} ({len(powB)} sessões), por banda de potência média."]
        if sig:
            lines.append("Bandas com diferença significativa (após correção de "
                         "múltiplas comparações, Holm): " + ", ".join(sig) + ".")
        else:
            lines.append("Nenhuma banda apresentou diferença significativa após a "
                         "correção de múltiplas comparações.")
        for band, r in results:
            lines.append(f"• {band}: {interpret_result(r)}")
        self.summary.setPlainText("\n".join(lines))
        return self._rows

    def _save(self):
        if not self._rows:
            QtWidgets.QMessageBox.information(
                self, "Nada a salvar", "Rode a comparação primeiro.")
            return
        start = getattr(self.main.config, "save_directory", "")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Salvar relatório",
            os.path.join(start, "estatistica_guiada.html"), "HTML (*.html)")
        if not path:
            return
        self._write_report(path)
        QtWidgets.QMessageBox.information(self, "Salvo", f"Relatório salvo:\n{path}")

    def _write_report(self, path):
        import csv as _csv
        csv_path = os.path.splitext(path)[0] + ".csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f); w.writerow(self.COLS)
            for r in self._rows:
                w.writerow([r[k] for k in self.COLS])
        nameA = self.name_A.text().strip() or "A"
        nameB = self.name_B.text().strip() or "B"
        rows_html = "".join(
            "<tr>" + "".join(f"<td>{r[k]}</td>" for k in self.COLS) + "</tr>"
            for r in self._rows)
        summ = self.summary.toPlainText().replace("&", "&amp;").replace(
            "<", "&lt;").replace("\n", "<br>")
        html = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Estatística guiada — {APP_NAME}</title><style>"
            "body{font-family:Arial,sans-serif;margin:24px;color:#1a2230}"
            "table{border-collapse:collapse;width:100%;font-size:14px}"
            "th,td{border:1px solid #dbe1ea;padding:6px 10px;text-align:left}"
            "th{background:#eef4f1;color:#0c7f5f}h1{color:#0f9d75;font-size:20px}"
            "</style></head><body><h1>Relatório de Estatística Guiada</h1>"
            f"<p>{APP_NAME} v{APP_VERSION} — {APP_AUTHORS}</p>"
            "<table><thead><tr>"
            f"<th>Banda</th><th>{nameA}</th><th>{nameB}</th><th>Teste</th>"
            "<th>p</th><th>p (Holm)</th><th>Efeito</th><th>Signif.?</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table>"
            f"<h3>Conclusão</h3><p>{summ}</p>"
            f"<p style='color:#5b6473;font-size:12px'>Gerado por {APP_NAME} "
            f"v{APP_VERSION} — {CODE_URL}</p></body></html>")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return csv_path


# ============================================================
# IntraSessionStatsDialog — comparar CONDIÇÕES dentro de uma sessão (genérico)
# ============================================================
class IntraSessionStatsDialog(QtWidgets.QDialog):
    """Facilitador GENÉRICO e MULTIMODAL: compara as condições/classes presentes
    em UMA sessão — sejam quais forem (dorsi/plantar, mãos, ou o que o usuário
    marcar) — por uma métrica que o próprio usuário escolhe (potência de banda
    para EEG, RMS para EMG/ECG, ou ERD% vs repouso). Não assume nenhum movimento."""

    METRICS = [("band", "Potência de banda (EEG)"),
               ("rms", "RMS / amplitude (qualquer sinal)"),
               ("erd", "ERD% vs repouso (EEG)")]

    def __init__(self, main, session=None):
        super().__init__(main)
        self.main = main
        self.d = None
        self._last = None
        self.setWindowTitle("Comparar condições da sessão (genérico/multimodal)")
        self.resize(840, 580)
        v = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Compara as <b>condições/classes marcadas nesta sessão</b> — sejam "
            "quais forem — pela métrica que você escolher. Genérico e multimodal: "
            "ajuste métrica/banda/canais ao seu caso (EEG, EMG, ECG, EoG). O teste "
            "estatístico é escolhido automaticamente.")
        intro.setWordWrap(True); intro.setTextFormat(QtCore.Qt.TextFormat.RichText)
        v.addWidget(intro)
        row0 = QtWidgets.QHBoxLayout()
        self.sess_lbl = QtWidgets.QLabel("(nenhuma sessão carregada)")
        load_btn = QtWidgets.QPushButton("Carregar sessão…")
        load_btn.clicked.connect(self._load)
        row0.addWidget(self.sess_lbl, 1); row0.addWidget(load_btn)
        v.addLayout(row0)
        rowm = QtWidgets.QHBoxLayout()
        rowm.addWidget(QtWidgets.QLabel("Métrica:"))
        self.metric_combo = QtWidgets.QComboBox()
        for key, lbl in self.METRICS:
            self.metric_combo.addItem(lbl, key)
        rowm.addWidget(self.metric_combo)
        rowm.addWidget(QtWidgets.QLabel("Banda:"))
        self.band_combo = QtWidgets.QComboBox()
        for b in EEG_BANDS:
            self.band_combo.addItem(b)
        rowm.addWidget(self.band_combo)
        rowm.addWidget(QtWidgets.QLabel("Canais:"))
        self.chan_combo = QtWidgets.QComboBox()
        self.chan_combo.addItem("Todos (média)")
        rowm.addWidget(self.chan_combo, 1)
        v.addLayout(rowm)
        run = QtWidgets.QPushButton("Comparar")
        run.clicked.connect(self._compute)
        v.addWidget(run)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Condição", "n", "Média ± DP", "Mediana"])
        self.table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.table, 1)
        self.summary = QtWidgets.QTextEdit(); self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(150)
        v.addWidget(self.summary)
        save = QtWidgets.QPushButton("Salvar relatório (HTML + CSV)…")
        save.clicked.connect(self._save)
        v.addWidget(save)
        if session:
            self.set_session(session)

    def set_session(self, d):
        self.d = d
        self.chan_combo.clear(); self.chan_combo.addItem("Todos (média)")
        for nm in d.get("ch_names", []):
            self.chan_combo.addItem(nm)
        dur = d["eeg"].shape[1] / d["sr"] if d.get("sr") else 0
        self.sess_lbl.setText(
            f"{d['eeg'].shape[0]} canais · {dur:.0f}s · "
            f"{len(d.get('trials', []))} eventos · "
            f"{len(d.get('markers', []))} marcadores")

    def _load(self):
        start = getattr(self.main.config, "save_directory", "")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Abrir sessão (data.csv)", start, "CSV (*.csv)")
        if not path:
            return
        d = self.main._load_session_csv(path)
        if d:
            self.set_session(d)

    def _chan_idx(self):
        i = self.chan_combo.currentIndex()
        n = self.d["eeg"].shape[0]
        if i <= 0:
            return list(range(n))
        return [min(i - 1, n - 1)]

    def _bandpower(self, seg, sr, lo, hi):
        vals = []
        for i in range(seg.shape[0]):
            f, psd = scipy_signal.welch(seg[i], fs=sr, nperseg=min(256, seg.shape[1]))
            m = (f >= lo) & (f < hi)
            vals.append(float(_TRAPEZOID(psd[m], f[m])) if np.any(m) else 0.0)
        return float(np.mean(vals)) if vals else 0.0

    def _conditions(self):
        """Genérico: {rótulo: [valor por trial/segmento]} + fonte. Usa trials
        (Event Id/Class Id) se houver; senão segmenta por marcadores."""
        d = self.d; eeg = d["eeg"]; sr = d["sr"]; N = eeg.shape[1]
        chans = self._chan_idx()
        metric = self.metric_combo.currentData()
        band = self.band_combo.currentText(); lo, hi = EEG_BANDS[band]
        base_phases = getattr(self.main, "BASELINE_PHASES",
                              ("baseline", "pre_rest", "inter_baseline"))

        def seg_of(s0, s1):
            s0 = max(0, int(s0)); s1 = min(N, int(s1))
            return eeg[np.ix_(chans, range(s0, s1))] if s1 - s0 >= 16 else None

        def value(seg, base_seg=None):
            if metric == "rms":
                return float(np.mean([np.sqrt(np.mean(seg[i] ** 2))
                                      for i in range(seg.shape[0])]))
            bp = self._bandpower(seg, sr, lo, hi)
            if metric == "erd":
                if base_seg is None:
                    return float("nan")
                bb = self._bandpower(base_seg, sr, lo, hi)
                return (bp - bb) / bb * 100.0 if bb > 0 else float("nan")
            return bp

        out = {}
        trials = [t for t in d.get("trials", []) if t.get("phase") == "mi"]
        if trials:
            base_trials = [t for t in d.get("trials", [])
                           if t.get("phase") in base_phases]
            for t in trials:
                seg = seg_of(t.get("start_line", 1) - 2, t.get("end_line", 1) - 1)
                if seg is None:
                    continue
                bseg = None
                if metric == "erd" and base_trials:
                    prev = [b for b in base_trials
                            if b.get("start_line", 0) <= t.get("start_line", 0)]
                    bt = prev[-1] if prev else base_trials[0]
                    bseg = seg_of(bt.get("start_line", 1) - 2,
                                  bt.get("end_line", 1) - 1)
                val = value(seg, bseg)
                if val is not None and np.isfinite(val):
                    cid = t.get("class_id", -1)
                    label = (t.get("class_name")
                             or (f"classe {cid}" if cid >= 0 else "MI"))
                    out.setdefault(str(label), []).append(val)
            # Adiciona REPOUSO como condição (permite comparar mesmo com 1 só
            # classe: "houve efeito vs repouso?"). ERD já é relativo à baseline,
            # então não entra como grupo.
            if metric != "erd":
                for b in base_trials:
                    seg = seg_of(b.get("start_line", 1) - 2, b.get("end_line", 1) - 1)
                    if seg is None:
                        continue
                    val = value(seg, None)
                    if val is not None and np.isfinite(val):
                        out.setdefault("Repouso", []).append(val)
            return out, "classes + repouso (Event Id)"
        # Fallback genérico: condições por marcadores (formato nativo)
        markers = d.get("markers", [])
        if markers:
            for i, (ts, lab) in enumerate(markers):
                t_next = markers[i + 1][0] if i + 1 < len(markers) else ts + 5.0
                seg = seg_of(ts * sr, min(t_next, ts + 5.0) * sr)
                if seg is None:
                    continue
                val = value(seg, seg if metric == "erd" else None)
                if val is not None and np.isfinite(val):
                    out.setdefault(str(lab), []).append(val)
            return out, "marcadores"
        return {}, "—"

    def _compute(self):
        if not self.d:
            QtWidgets.QMessageBox.warning(self, "Sessão",
                                         "Carregue uma sessão primeiro.")
            return None
        groups, source = self._conditions()
        groups = {k: vv for k, vv in groups.items() if len(vv) >= 1}
        if len(groups) < 2:
            QtWidgets.QMessageBox.information(
                self, "Condições insuficientes",
                "Não encontrei pelo menos 2 condições/classes nesta sessão. Use "
                "uma sessão com eventos/marcadores (ou compare grupos de sessões "
                "em 'Estatística guiada').")
            return None
        names = list(groups.keys()); vals = [groups[k] for k in names]
        r = auto_compare(vals, names)
        self.table.setRowCount(len(names))
        rows = []
        for i, nm in enumerate(names):
            dsc = r["descr"][i]
            cells = [nm, str(dsc["n"]),
                     f"{dsc['mean']:.3g} ± {dsc['sd']:.2g}",
                     f"{dsc['median']:.3g}"]
            for j, c in enumerate(cells):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(c))
            rows.append(cells)
        self.table.resizeColumnsToContents()
        metric_lbl = dict(self.METRICS)[self.metric_combo.currentData()]
        band_txt = (f" ({self.band_combo.currentText()})"
                    if self.metric_combo.currentData() != "rms" else "")
        head = (f"Métrica: {metric_lbl}{band_txt}  |  Canais: "
                f"{self.chan_combo.currentText()}  |  Condições de: {source}")
        concl = interpret_result(r)
        self.summary.setPlainText(head + "\n\n" + concl)
        self._last = {"rows": rows, "head": head, "concl": concl}
        return self._last

    def _save(self):
        if not self._last:
            QtWidgets.QMessageBox.information(self, "Nada a salvar",
                                             "Rode a comparação primeiro.")
            return
        start = getattr(self.main.config, "save_directory", "")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Salvar relatório",
            os.path.join(start, "comparacao_condicoes.html"), "HTML (*.html)")
        if not path:
            return
        import csv as _csv
        csv_path = os.path.splitext(path)[0] + ".csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f); w.writerow(["condicao", "n", "media_dp", "mediana"])
            for r in self._last["rows"]:
                w.writerow(r)
        rows_html = "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
            for r in self._last["rows"])
        summ = (self._last["head"] + "\n\n" + self._last["concl"]).replace(
            "&", "&amp;").replace("<", "&lt;").replace("\n", "<br>")
        html = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Comparação de condições — {APP_NAME}</title><style>"
            "body{font-family:Arial,sans-serif;margin:24px;color:#1a2230}"
            "table{border-collapse:collapse;width:100%;font-size:14px}"
            "th,td{border:1px solid #dbe1ea;padding:6px 10px;text-align:left}"
            "th{background:#eef4f1;color:#0c7f5f}h1{color:#0f9d75;font-size:20px}"
            "</style></head><body><h1>Comparação de condições da sessão</h1>"
            f"<p>{APP_NAME} v{APP_VERSION} — {APP_AUTHORS}</p>"
            "<table><thead><tr><th>Condição</th><th>n</th><th>Média ± DP</th>"
            f"<th>Mediana</th></tr></thead><tbody>{rows_html}</tbody></table>"
            f"<h3>Conclusão</h3><p>{summ}</p>"
            f"<p style='color:#5b6473;font-size:12px'>Gerado por {APP_NAME} "
            f"v{APP_VERSION} — {CODE_URL}</p></body></html>")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        QtWidgets.QMessageBox.information(self, "Salvo", f"Relatório salvo:\n{path}")


# ============================================================
# SerialReaderThread — Hardware/Simulação/Playback com suporte a expansão 16ch
# ============================================================
class SerialReaderThread(QThread):
    data_received    = Signal(np.ndarray, np.ndarray)
    error            = Signal(str)
    connection_state = Signal(bool)
    progress         = Signal(float)
    expansion_detected = Signal(int)  # emite num de canais detectados (8 ou 16)

    SCALE_UV    = 4.5 / 24.0 / (2 ** 23 - 1) * 1_000_000.0
    SCALE_ACCEL = 0.002

    def __init__(self, port=None, baud_rate=115200, mode="hardware",
                 playback_path=None, daisy=False, num_channels=None, parent=None):
        super().__init__(parent)
        self.port = port
        self.baud_rate = baud_rate
        self.mode = mode
        self.playback_path = playback_path
        self.daisy = bool(daisy)
        # Aceita override explícito de num_channels (para suportar 24/32/40/48/56/64
        # vindos da placa customizada). Se ausente, cai no comportamento legacy
        # 8 ou 16 baseado no flag `daisy` (compat com chamadas antigas).
        if num_channels is None:
            self.num_channels = CYTON_MAX_CHANNELS if self.daisy else BASE_CHANNELS
        else:
            self.num_channels = max(BASE_CHANNELS, min(MAX_CHANNELS, int(num_channels)))
        self._serial = None
        self._running = False
        self._daisy_upper = None   # buffer da metade "superior" (8ch) entre pacotes Cyton

    def run(self):
        if self.mode == "simulation":
            self._run_simulation()
        elif self.mode == "playback":
            self._run_playback()
        else:
            self._run_serial()

    def _run_simulation(self):
        self.connection_state.emit(True)
        self._running = True
        t = 0.0
        dt = 1.0 / SAMPLE_RATE
        rng = np.random.default_rng()
        accel = np.zeros(3)
        n = self.num_channels
        while self._running:
            sample = np.empty(n)
            for ch in range(n):
                a  = 30.0 * np.sin(2 * np.pi * 10 * t + ch * 0.5)
                b  = 10.0 * np.sin(2 * np.pi * 20 * t + ch * 0.3)
                th = 20.0 * np.sin(2 * np.pi *  6 * t + ch * 0.7)
                noise = rng.normal(0, 8.0)
                sample[ch] = a + b + th + noise
            accel[0] = 0.05 * np.sin(2 * np.pi * 0.3 * t)
            accel[1] = 0.05 * np.cos(2 * np.pi * 0.3 * t)
            accel[2] = 1.00 + 0.02 * np.sin(2 * np.pi * 2 * t)
            self.data_received.emit(sample, accel)
            t += dt
            time.sleep(dt * 0.95)
        self.connection_state.emit(False)

    def _run_playback(self):
        if not self.playback_path or not os.path.exists(self.playback_path):
            self.error.emit(f"CSV de playback não encontrado: {self.playback_path}")
            return
        try:
            # Le o cabecalho para NAO tentar converter a coluna textual 'marker'
            # (data.csv nativo sempre termina em 'marker' -> np.loadtxt falharia
            #  ate quando os marcadores estao vazios). Espelha _load_native_session_csv.
            usecols = None
            try:
                with open(self.playback_path, "r", encoding="utf-8", errors="ignore") as _f:
                    header = next(csv.reader(_f))
                num_cols = [i for i, h in enumerate(header)
                            if h.strip().endswith("_uV")
                            or h.strip() in ("timestamp_s", "ax_g", "ay_g", "az_g")]
                if num_cols:
                    usecols = num_cols
            except Exception:
                usecols = None
            data = np.loadtxt(self.playback_path, delimiter=",", skiprows=1,
                              usecols=usecols, encoding="utf-8")
        except Exception as exc:
            self.error.emit(f"Falha lendo CSV: {exc}")
            return
        if data.ndim == 1:
            data = data.reshape(1, -1)
        n_rows, n_cols = data.shape

        # Auto-detecção multi-step do número de canais a partir das colunas do CSV
        # Formato esperado: timestamp + N canais + ax + ay + az [+ marker]
        # Tenta encaixar nos passos válidos: 8, 16, 24, 32, 40, 48, 56, 64
        inferred = n_cols - 1 - 3  # remove timestamp + 3 accel; sem assumir marker
        detected = None
        for cand in (inferred, inferred - 1):  # tenta com e sem marker
            if cand in EXPANSION_STEPS:
                detected = cand; break
        if detected is None:
            # Snap ao passo mais próximo dentro do range válido
            if BASE_CHANNELS <= inferred <= MAX_CHANNELS:
                detected = min(EXPANSION_STEPS, key=lambda s: abs(s - inferred))
            else:
                detected = max(1, n_cols - 1)
        if detected != self.num_channels:
            self.num_channels = detected
            self.expansion_detected.emit(detected)

        self.connection_state.emit(True)
        self._running = True
        dt = 1.0 / SAMPLE_RATE
        for i in range(n_rows):
            if not self._running: break
            row = data[i]
            if row.shape[0] >= 1 + detected:
                sample = row[1 : 1 + detected]
            else:
                sample = np.zeros(detected)
            if row.shape[0] >= 1 + detected + 3:
                accel = row[1 + detected : 1 + detected + 3]
            else:
                accel = np.zeros(3)
            self.data_received.emit(sample, accel)
            if (i & 0x3F) == 0:
                self.progress.emit((i + 1) / n_rows)
            time.sleep(dt * 0.95)
        self.progress.emit(1.0)
        self._running = False
        self.connection_state.emit(False)

    def _run_serial(self):
        try:
            self._serial = serial.Serial(self.port, self.baud_rate, timeout=1.0)
            time.sleep(2.0)
            try:
                self._serial.reset_input_buffer()
                # Se expansão solicitada, manda o comando C antes de iniciar streaming
                if self.daisy:
                    self._serial.write(b"C")
                    time.sleep(0.4)
                self._serial.write(b"b")
            except Exception: pass
        except (serial.SerialException, OSError) as exc:
            self.error.emit(f"Falha ao abrir {self.port}: {exc}")
            return
        self.connection_state.emit(True)
        self._running = True
        try:
            while self._running:
                packet = self._read_packet()
                if packet is not None:
                    sample, accel = packet
                    self.data_received.emit(sample, accel)
        except Exception as exc:
            self.error.emit(f"Erro durante leitura: {exc}")
        finally:
            self._close_serial()
            self.connection_state.emit(False)

    def _read_packet(self):
        """Le um pacote do protocolo. Em modo expansão, intercala dois pacotes para
        formar um sample de 16 canais."""
        head = self._serial.read(1)
        if not head or head[0] != 0xA0:
            return None
        payload = self._serial.read(32)
        if len(payload) < 32:
            return None
        # Decodifica 8 canais EEG
        eight = np.empty(BASE_CHANNELS)
        for ch in range(BASE_CHANNELS):
            base = 1 + ch * 3
            raw = (payload[base] << 16) | (payload[base + 1] << 8) | payload[base + 2]
            if raw & 0x800000:
                raw -= 1 << 24
            eight[ch] = raw * self.SCALE_UV
        # Acelerômetro
        accel = np.zeros(3)
        try:
            if payload[31] == 0xC0:
                for k in range(3):
                    raw = int.from_bytes(payload[25 + k*2 : 27 + k*2], "big", signed=True)
                    accel[k] = raw * self.SCALE_ACCEL
        except Exception: pass

        if not self.daisy:
            return eight, accel

        # Em modo expansão, intercala pacotes. Sample number byte = payload[0]
        # Convenção do protocolo: bit alto indica board "upper" vs "lower"
        sample_num = payload[0]
        is_upper = (sample_num & 0x01) == 0  # heurica: par = upper, impar = lower
        if is_upper:
            self._daisy_upper = eight
            return None  # ainda não temos o par
        else:
            if self._daisy_upper is None:
                return None
            sixteen = np.concatenate([self._daisy_upper, eight])
            self._daisy_upper = None
            return sixteen, accel

    def send_command(self, cmd):
        if self._serial and self._serial.is_open:
            try:
                self._serial.write(cmd.encode("ascii"))
                return True
            except Exception:
                return False
        return False

    def _close_serial(self):
        if self._serial and self._serial.is_open:
            try: self._serial.write(b"s")
            except: pass
            try: self._serial.close()
            except: pass
        self._serial = None

    def stop(self):
        self._running = False
        self.wait(2500)


# ============================================================
# Leitor EDF/BDF TOLERANTE (numpy puro) — lê exames clínicos mesmo com
# cabeçalho quebrado (ex.: iCelera com acento no nome do paciente, que faz
# EDFbrowser/pyedflib recusarem o arquivo inteiro). Só lê os campos NUMÉRICOS
# do header; ignora textos não-ASCII. Não depende de pyedflib (funciona no .exe).
# Especificação EDF: https://www.edfplus.info/specs/edf.html
# ============================================================
def read_edf(path):
    """Lê um EDF/BDF de forma tolerante. Retorna dict com labels, signals
    (µV físicos), fs por canal, etc. Ignora o campo de texto do paciente."""
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 256:
        raise ValueError("Arquivo muito curto para ser EDF/BDF.")
    H = raw[:256]

    def _num(b, default="0"):
        s = b.decode("ascii", "ignore").strip()
        return s or default

    try:
        ns = int(_num(H[252:256]))
    except ValueError:
        raise ValueError("Cabeçalho EDF inválido (nº de sinais).")
    if ns <= 0 or ns > 2048:
        raise ValueError(f"Nº de sinais fora do esperado: {ns}")
    n_records = int(_num(H[236:244], "-1"))
    rec_dur = float(_num(H[244:252], "1")) or 1.0
    reserved = H[192:236].decode("ascii", "replace").upper()
    is_bdf = (raw[:1] == b"\xff") or ("BDF" in reserved) or path.lower().endswith(".bdf")

    p = [256]
    def field(size):
        chunk = raw[p[0]:p[0] + size * ns]; p[0] += size * ns
        return [chunk[i * size:(i + 1) * size] for i in range(ns)]

    labels   = [x.decode("ascii", "replace").strip() for x in field(16)]
    field(80)                                       # transducer
    phys_dim = [x.decode("ascii", "replace").strip() for x in field(8)]
    phys_min = [float(_num(x)) for x in field(8)]
    phys_max = [float(_num(x)) for x in field(8)]
    dig_min  = [float(_num(x)) for x in field(8)]
    dig_max  = [float(_num(x)) for x in field(8)]
    field(80)                                       # prefiltering
    nsamp    = [int(_num(x)) for x in field(8)]     # amostras por registro
    field(32)                                       # reserved por sinal

    bps = 3 if is_bdf else 2
    data_start = 256 + ns * 256
    rec_size = int(np.sum(nsamp)) * bps
    if n_records < 0 and rec_size:
        n_records = max(0, (len(raw) - data_start) // rec_size)
    sig_off = (np.concatenate([[0], np.cumsum(nsamp)]) * bps).astype(np.int64)
    sigs = [np.empty(nsamp[s] * n_records, dtype=np.float64) for s in range(ns)]
    for r in range(n_records):
        base = data_start + r * rec_size
        for s in range(ns):
            cnt = nsamp[s]
            if cnt == 0:
                continue
            st = base + int(sig_off[s])
            b = raw[st:st + cnt * bps]
            if len(b) < cnt * bps:
                cnt = len(b) // bps
                b = b[:cnt * bps]
            if is_bdf:
                a = np.frombuffer(b, np.uint8).reshape(-1, 3).astype(np.int32)
                v = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
                v = np.where(v >= (1 << 23), v - (1 << 24), v).astype(np.float64)
            else:
                v = np.frombuffer(b, "<i2").astype(np.float64)
            sigs[s][r * nsamp[s]: r * nsamp[s] + cnt] = v
    for s in range(ns):
        dmn, dmx, pmn, pmx = dig_min[s], dig_max[s], phys_min[s], phys_max[s]
        if dmx != dmn:
            sigs[s] = pmn + (sigs[s] - dmn) * (pmx - pmn) / (dmx - dmn)
    fs = [nsamp[s] / rec_dur if rec_dur else 0.0 for s in range(ns)]
    return {"labels": labels, "signals": sigs, "fs": fs, "phys_dim": phys_dim,
            "n_records": n_records, "rec_dur": rec_dur, "is_bdf": is_bdf}


def edf_to_native_csv(edf_path, out_csv, max_seconds=None):
    """Converte um EDF/BDF para o CSV NATIVO do OpenBiônica (time_s,<ch>_uV,...)
    para abrir direto no modo Offline. Descarta canais de anotação e usa a taxa
    de amostragem dominante (canais EEG). Retorna (out_csv, labels, fs, n)."""
    r = read_edf(edf_path)
    idxs = [i for i, l in enumerate(r["labels"])
            if "annotation" not in l.lower() and r["fs"][i] > 0 and len(r["signals"][i])]
    if not idxs:
        raise ValueError("Nenhum canal de sinal válido encontrado no EDF.")
    # taxa de amostragem dominante (moda dos fs arredondados)
    fs_round = [int(round(r["fs"][i])) for i in idxs]
    fs = max(set(fs_round), key=fs_round.count)
    keep = [i for i in idxs if int(round(r["fs"][i])) == fs]
    labels = [r["labels"][i] or f"CH{k+1}" for k, i in enumerate(keep)]
    n = min(len(r["signals"][i]) for i in keep)
    if max_seconds:
        n = min(n, int(max_seconds * fs))
    t = (np.arange(n) / float(fs)).reshape(-1, 1)
    mat = np.column_stack([r["signals"][i][:n] for i in keep])
    data = np.column_stack([t, mat])
    header = "time_s," + ",".join(f"{l}_uV" for l in labels)
    # tempo com precisão fixa (senão a SR medida sai errada em exames longos);
    # sinais em 6 algarismos significativos (mais que suficiente p/ µV).
    fmt = ["%.6f"] + ["%.6g"] * len(labels)
    np.savetxt(out_csv, data, delimiter=",", header=header, comments="", fmt=fmt)
    return out_csv, labels, fs, n


# ============================================================
# ICA (FastICA) em numpy PURO — remove artefato ocular (piscada) sem depender
# do MNE nem de pip install. Funciona no .exe. Algoritmo FastICA simétrico
# (Hyvärinen 1999) com não-linearidade tanh.
# ============================================================
def _fast_ica_W(Xw, n_components, max_iter=250, tol=1e-5, seed=42):
    """FastICA simétrico sobre dados JÁ branqueados Xw (n_comp x n)."""
    n = Xw.shape[1]
    rng = np.random.RandomState(seed)
    def _sym(W):
        u, _s, vt = np.linalg.svd(W, full_matrices=False)
        return u @ vt
    W = _sym(rng.randn(n_components, n_components))
    for _ in range(max_iter):
        WX = W @ Xw
        g = np.tanh(WX)
        gp = (1.0 - g * g).mean(axis=1)
        Wn = _sym((g @ Xw.T) / n - gp[:, None] * W)
        lim = np.max(np.abs(np.abs(np.sum(Wn * W, axis=1)) - 1.0))
        W = Wn
        if lim < tol:
            break
    return W


def ica_clean_eog(eeg_uV, sr, ch_names, n_components=None):
    """Remove piscadas por ICA (numpy puro). Filtra 1-40 Hz (estabiliza a ICA),
    detecta o componente ocular (correlação com canais frontais ou curtose) e
    reconstrói. Retorna (eeg_limpo_uV, indices_excluidos, info)."""
    X = np.asarray(eeg_uV, dtype=float)
    n_ch, n = X.shape
    if n_ch < 2 or n < 32:
        return X, [], {"backend": "numpy", "note": "sinal curto demais"}
    try:
        hi = min(40.0, sr / 2.0 * 0.98)
        sos = scipy_signal.butter(4, [1.0, hi], btype="band", fs=sr, output="sos")
        Xf = scipy_signal.sosfiltfilt(sos, X, axis=1)
    except Exception:
        Xf = X - X.mean(axis=1, keepdims=True)
    mean = Xf.mean(axis=1, keepdims=True)
    Xc = Xf - mean
    cov = np.cov(Xc)
    dvals, E = np.linalg.eigh(cov)
    order = np.argsort(dvals)[::-1]
    k = min(n_components or 15, n_ch)
    sel = order[:k]
    dsel = np.maximum(dvals[sel], 1e-12)
    K = (E[:, sel] / np.sqrt(dsel)).T          # branqueamento  k x n_ch
    Xw = K @ Xc
    W = _fast_ica_W(Xw, k)
    S = W @ Xw                                  # fontes  k x n
    frontal = [i for i, nm in enumerate(ch_names)
               if nm.upper().replace("EEG", "").replace("-", "").strip()
               in ("FP1", "FP2", "FPZ", "AF7", "AF8", "FP1F", "FP2F")]
    excluded = []
    if frontal:
        fsig = Xc[frontal].mean(axis=0)
        cors = np.array([abs(np.corrcoef(S[c], fsig)[0, 1]) for c in range(k)])
        cmax = int(np.nanargmax(cors))
        if np.isfinite(cors[cmax]) and cors[cmax] > 0.5:
            excluded = [cmax]
    if not excluded:                            # fallback: fonte mais impulsiva
        kurt = np.array([float(((S[c] - S[c].mean()) ** 4).mean()
                               / (S[c].var() ** 2 + 1e-12) - 3.0) for c in range(k)])
        cmax = int(np.argmax(kurt))
        if kurt[cmax] > 10:
            excluded = [cmax]
    S2 = S.copy()
    for c in excluded:
        S2[c] = 0.0
    clean = (np.linalg.pinv(K) @ (np.linalg.pinv(W) @ S2)) + mean
    return clean, excluded, {"backend": "numpy", "n_components": k}


# ============================================================
# SignalProcessor
# ============================================================
class SignalProcessor:
    @staticmethod
    def compute_fft(data, sample_rate=SAMPLE_RATE):
        n = len(data)
        if n < 2:
            return np.array([]), np.array([])
        window = np.hanning(n)
        # 2/sum(window) compensa o ganho coerente da janela (Hanning ~0.5);
        # com janela retangular sum=n, generaliza. Antes usava 2/n -> ~2x baixo.
        wsum = float(np.sum(window)) or float(n)
        spectrum = np.abs(rfft(data * window)) * 2.0 / wsum
        freqs = rfftfreq(n, 1.0 / sample_rate)
        return freqs, spectrum

    @staticmethod
    def compute_band_powers(data, sample_rate=SAMPLE_RATE):
        if len(data) < sample_rate:
            return {b: 0.0 for b in EEG_BANDS}
        nperseg = min(256, len(data))
        freqs, psd = scipy_signal.welch(data, fs=sample_rate, nperseg=nperseg)
        powers = {}
        for band, (low, high) in EEG_BANDS.items():
            mask = (freqs >= low) & (freqs < high)
            powers[band] = float(_TRAPEZOID(psd[mask], freqs[mask])) if np.any(mask) else 0.0
        return powers

    @staticmethod
    def compute_statistics(data):
        if len(data) == 0:
            return {"mean": 0.0, "std": 0.0, "rms": 0.0}
        return {
            "mean": float(np.mean(data)),
            "std":  float(np.std(data)),
            "rms":  float(np.sqrt(np.mean(np.square(data)))),
        }

    @staticmethod
    def compute_psd_column(data, sample_rate=SAMPLE_RATE, fmax=SPEC_FMAX, n_bins=SPEC_FMAX):
        if len(data) < sample_rate:
            return np.full(n_bins, -80.0)
        nperseg = min(256, len(data))
        freqs, psd = scipy_signal.welch(data, fs=sample_rate, nperseg=nperseg)
        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-12))
        grid = np.linspace(0.0, fmax, n_bins, endpoint=False)
        return np.interp(grid, freqs, psd_db)

    @staticmethod
    def compute_band_power(data, low, high, sample_rate=SAMPLE_RATE):
        if len(data) < sample_rate:
            return 0.0
        nperseg = min(256, len(data))
        freqs, psd = scipy_signal.welch(data, fs=sample_rate, nperseg=nperseg)
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask): return 0.0
        return float(_TRAPEZOID(psd[mask], freqs[mask]))

    @staticmethod
    def compute_focus_index(data, sample_rate=SAMPLE_RATE):
        alpha = SignalProcessor.compute_band_power(data, 8.0, 13.0, sample_rate)
        beta  = SignalProcessor.compute_band_power(data, 13.0, 30.0, sample_rate)
        s = alpha + beta
        if s < 1e-9: return 0.5
        return float(beta / s)

    @staticmethod
    def compute_emg_envelope(data, sample_rate=SAMPLE_RATE, window_s=0.10):
        if len(data) < 2:
            return np.zeros_like(data)
        w = max(1, int(sample_rate * window_s))
        absdata = np.abs(data)
        if w >= len(absdata):
            return np.full_like(absdata, np.mean(absdata))
        kernel = np.ones(w) / w
        return np.convolve(absdata, kernel, mode="same")

    # ====================================================================
    # ERS/ERD — Event-Related (De)Synchronization
    # Método de Pfurtscheller & Lopes da Silva (1999).
    # Para cada trial: P_event = potência banda durante a fase MI.
    #                  P_baseline = potência banda durante a fase baseline.
    # ERD%(canal) = ((P_event - P_baseline) / P_baseline) × 100
    # Negativo = dessincronização (típico em motor imagery sobre córtex motor
    # contralateral). Positivo = sincronização ("beta rebound" pós-movimento).
    # ====================================================================
    @staticmethod
    def compute_ersd_per_channel(eeg, sr, trials_mi, trials_baseline_ref,
                                  band_low, band_high):
        """Para cada canal, calcula ERD/ERS médio entre trials.
        Args:
          eeg: array (n_ch, n_samp), em µV
          sr:  sample rate
          trials_mi: lista de (start_line, end_line) das janelas MI (1-indexed
                     conforme o events.csv do BCI system)
          trials_baseline_ref: lista de (start_line, end_line) das janelas
                     de referência (baseline ou pre_rest) para cada trial
          band_low, band_high: faixa em Hz
        Retorna: np.array (n_ch,) com ERD% (negativo = dessinc.)
        """
        n_ch = eeg.shape[0]
        out = np.zeros(n_ch)
        if not trials_mi or not trials_baseline_ref: return out
        for ch in range(n_ch):
            ratios = []
            for (mi_s, mi_e), (bl_s, bl_e) in zip(trials_mi,
                                                    trials_baseline_ref):
                # csv_start_line/csv_end_line são 1-indexed nas linhas DE DADOS
                # (ou seja, sample_index = line - 1 se o header já foi subtraído).
                # Nas saídas dos dois sistemas, linha 1 = primeira amostra; o
                # header é parte separada. Vamos clamp e usar como sample idx.
                a = max(0, mi_s - 1); b = min(eeg.shape[1], mi_e)
                c = max(0, bl_s - 1); d = min(eeg.shape[1], bl_e)
                if b - a < int(sr * 0.5) or d - c < int(sr * 0.5):
                    continue   # menos de 0.5s, ignora
                p_mi = SignalProcessor.compute_band_power(
                    eeg[ch, a:b], band_low, band_high, sr)
                p_bl = SignalProcessor.compute_band_power(
                    eeg[ch, c:d], band_low, band_high, sr)
                if p_bl < 1e-12: continue
                ratios.append((p_mi - p_bl) / p_bl * 100.0)
            if ratios:
                out[ch] = float(np.mean(ratios))
        return out

    @staticmethod
    def compute_ersd_timecourse(eeg, sr, trials_mi, baseline_pwr_per_ch,
                                  band_low, band_high, window_s=0.5,
                                  step_s=0.1, lock_before_s=1.0,
                                  duration_s=4.0):
        """Curso temporal ERD%(t) médio entre trials, com janela deslizante.
        Para cada trial, computa banda em janelas (window_s, step_s) cobrindo
        [-lock_before_s, +duration_s] em torno do início do MI. Normaliza
        cada trial pela potência baseline daquele canal.
        Retorna (t_axis (T,), ersd (n_ch, T))
        """
        n_ch = eeg.shape[0]
        win_n  = int(round(window_s * sr))
        step_n = int(round(step_s * sr))
        if win_n < 4 or step_n < 1:
            return np.array([]), np.zeros((n_ch, 0))
        before_n = int(round(lock_before_s * sr))
        after_n  = int(round(duration_s * sr))
        # gera centros (em samples) relativos ao início do MI
        centers = np.arange(-before_n + win_n // 2,
                             after_n - win_n // 2 + 1, step_n)
        if centers.size == 0:
            return np.array([]), np.zeros((n_ch, 0))
        t_axis = centers / sr
        T = len(centers)
        accum = np.zeros((n_ch, T)); cnt = np.zeros((n_ch, T))
        for ti, (mi_s, _mi_e) in enumerate(trials_mi):
            mi_start = mi_s - 1  # 1-indexed → 0-indexed
            for k, c in enumerate(centers):
                a = mi_start + c - win_n // 2
                b = a + win_n
                if a < 0 or b > eeg.shape[1]: continue
                for ch in range(n_ch):
                    p = SignalProcessor.compute_band_power(
                        eeg[ch, a:b], band_low, band_high, sr)
                    p_base = baseline_pwr_per_ch[ch] if ch < len(baseline_pwr_per_ch) else 1e-9
                    if p_base < 1e-12: continue
                    accum[ch, k] += (p - p_base) / p_base * 100.0
                    cnt[ch, k]   += 1
        ersd = np.zeros((n_ch, T))
        mask = cnt > 0
        ersd[mask] = accum[mask] / cnt[mask]
        return t_axis, ersd


# ============================================================
# HeadPlotWidget — suporta 8 ou 16 eletrodos
# ============================================================
class HeadPlotWidget(QtWidgets.QWidget):
    """Head plot com 3 modos visuais:
       1) Eletrodos coloridos por potência (dots)
       2) Mapa de calor INTERPOLADO (IDW) sobre o couro cabeludo
       3) Ambos sobrepostos (default)
    O mapeamento CH -> nome de eletrodo eh dinâmico (do AppConfig)."""

    HEATMAP_GRID = 96  # resolucao do mapa de calor (NxN)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.powers = [0.0] * MAX_CHANNELS
        self.raw_values = [0.0] * MAX_CHANNELS
        self.num_channels = BASE_CHANNELS
        self.band_name = "Alpha"
        self.show_heatmap = True
        self.show_dots = True
        self.map_mode = "interp"   # "interp" (potencial) | "csd" (Laplaciano)
        # Mapeamento default; sera atualizado por set_mapping()
        self.mapping = list(DEFAULT_MAPPING)
        self.setMinimumSize(420, 420)
        self.setStyleSheet(f"background-color: {COLORS['surface']};")

    def set_num_channels(self, n):
        self.num_channels = n
        self.update()

    def set_mapping(self, mapping):
        """Define qual eletrodo (do 10-20) cada canal CHn ocupa."""
        if isinstance(mapping, (list, tuple)) and len(mapping) == MAX_CHANNELS:
            self.mapping = list(mapping)
            self.update()

    def set_show_heatmap(self, on):
        self.show_heatmap = bool(on); self.update()

    def set_show_dots(self, on):
        self.show_dots = bool(on); self.update()

    def set_powers(self, values, band_name="Alpha"):
        self.raw_values = [float(v) for v in values]
        self.band_name = band_name
        self._recompute_display()
        self.update()

    def set_map_mode(self, mode):
        """Modo do mapa: 'interp' (potencial de escalpo, IDW) ou 'csd'
        (Laplaciano de superfície — realça fontes locais, reduz condução de
        volume [McFarland 1997; Nunez & Srinivasan 2006])."""
        self.map_mode = "csd" if str(mode).lower().startswith("csd") else "interp"
        self._recompute_display()
        self.update()

    def _recompute_display(self):
        """Recalcula os valores 0..1 exibidos, conforme o modo (interp/CSD)."""
        n = max(0, int(self.num_channels))
        raw = np.array(self.raw_values[:n], dtype=float) if n else np.array([])
        if self.map_mode == "csd" and n >= 4:
            pos = np.array([ALL_ELECTRODES.get(
                self.mapping[i] if i < len(self.mapping) else None, (0.0, 0.0))
                for i in range(n)], dtype=float)
            csd = raw.copy()
            k = min(4, n - 1)
            for i in range(n):
                d2 = np.sum((pos - pos[i]) ** 2, axis=1); d2[i] = np.inf
                nn = np.argsort(d2)[:k]
                csd[i] = raw[i] - float(np.mean(raw[nn]))
            vals = np.abs(csd)                       # magnitude da fonte local
        else:
            vals = np.clip(raw, 0.0, None)
        vmax = float(np.max(vals)) if vals.size and np.max(vals) > 0 else 1.0
        disp = list(np.clip(vals / vmax, 0.0, 1.0)) if vals.size else []
        self.powers = disp + [0.0] * (MAX_CHANNELS - len(disp))

    # ----- interpolacao IDW (vetorizada) ---------------------------
    def _compute_heatmap_image(self, r_px):
        """Gera QImage NxN com mapa de calor interpolado dentro do disco."""
        N = self.HEATMAP_GRID
        # coordenadas normalizadas (-1..1)
        ys, xs = np.mgrid[-1:1:N*1j, -1:1:N*1j]
        # mascara do disco
        mask = (xs**2 + ys**2) <= 1.0

        # pontos dos eletrodos ativos
        pts, vals = [], []
        for i in range(self.num_channels):
            name = self.mapping[i] if i < len(self.mapping) else None
            if name in ALL_ELECTRODES:
                ex, ey = ALL_ELECTRODES[name]
                pts.append((ex, -ey))   # y invertido para coords de imagem
                vals.append(self.powers[i])
        if not pts:
            img = QtGui.QImage(N, N, QtGui.QImage.Format.Format_ARGB32)
            img.fill(QtGui.QColor(0, 0, 0, 0))
            return img
        pts = np.array(pts);  vals = np.array(vals)

        # IDW: w_i = 1 / d_i^2
        gxy = np.stack([xs.ravel(), ys.ravel()], axis=1)            # (N*N, 2)
        diff = gxy[:, None, :] - pts[None, :, :]                     # (N*N, K, 2)
        d2 = np.maximum(np.sum(diff**2, axis=2), 1e-6)               # (N*N, K)
        w = 1.0 / d2
        interp = np.sum(w * vals[None, :], axis=1) / np.sum(w, axis=1)
        interp = interp.reshape(N, N)
        interp = np.clip(interp, 0.0, 1.0)

        # constroi imagem ARGB
        img = QtGui.QImage(N, N, QtGui.QImage.Format.Format_ARGB32)
        img.fill(QtGui.QColor(0, 0, 0, 0))
        for j in range(N):
            for i in range(N):
                if mask[j, i]:
                    color = self._color_for_power(interp[j, i])
                    color.setAlpha(220)
                    img.setPixelColor(i, j, color)
        return img

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0 + 8
        r = min(w, h) / 2.0 - 50

        p.fillRect(0, 0, w, h, QtGui.QColor(COLORS["surface"]))

        # ----- Mapa de calor interpolado (atras dos eletrodos) -----
        if self.show_heatmap and self.num_channels > 0:
            try:
                hm = self._compute_heatmap_image(r)
                # desenha esticado no disco
                p.save()
                clip = QtGui.QPainterPath()
                clip.addEllipse(QtCore.QPointF(cx, cy), r, r)
                p.setClipPath(clip)
                p.drawImage(QtCore.QRectF(cx - r, cy - r, 2*r, 2*r), hm)
                p.restore()
            except Exception:
                pass

        # Contorno da cabeça + nariz + orelhas
        p.setPen(QtGui.QPen(QtGui.QColor(COLORS["text_dim"]), 2))
        p.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 0)))
        p.drawEllipse(QtCore.QPointF(cx, cy), r, r)
        nose_h = 14
        path = QtGui.QPainterPath()
        path.moveTo(cx, cy - r - nose_h)
        path.lineTo(cx - 10, cy - r + 4)
        path.lineTo(cx + 10, cy - r + 4)
        path.closeSubpath()
        p.setBrush(QtGui.QBrush(QtGui.QColor(COLORS["surface_alt"])))
        p.drawPath(path)
        p.drawEllipse(QtCore.QPointF(cx - r - 4, cy), 7, 14)
        p.drawEllipse(QtCore.QPointF(cx + r + 4, cy), 7, 14)

        # Titulo (interface — Inter)
        p.setPen(QtGui.QPen(QtGui.QColor(COLORS["accent"])))
        title_font = QtGui.QFont(FONT_UI, 11, QtGui.QFont.Weight.Bold)
        p.setFont(title_font)
        mode_txt = "CSD/LAPLACIANO" if self.map_mode == "csd" else "POTÊNCIA"
        p.drawText(QtCore.QRectF(0, 4, w, 18), QtCore.Qt.AlignmentFlag.AlignCenter,
                   f"{mode_txt} · {self.band_name.upper()}  ({self.num_channels}ch)")

        # ----- Eletrodos (dots) -----
        if self.show_dots:
            radius = 18 if self.num_channels > 8 else 20
            # Nomes dos eletrodos (Fp1, Cz, etc) — Inter (interface)
            name_font = QtGui.QFont(FONT_UI, 8 if self.num_channels > 8 else 9,
                                    QtGui.QFont.Weight.Bold)
            # Valores numericos — JetBrains Mono (dados)
            value_font = QtGui.QFont(FONT_DATA, 7, QtGui.QFont.Weight.Normal)
            for i in range(self.num_channels):
                name = self.mapping[i] if i < len(self.mapping) else f"CH{i+1}"
                pos = ALL_ELECTRODES.get(name)
                if pos is None:
                    continue
                nx, ny = pos
                ex = cx + nx * r * 0.92
                ey = cy - ny * r * 0.92
                color = self._color_for_power(self.powers[i])
                p.setBrush(QtGui.QBrush(color))
                border_color = QtGui.QColor(COLORS["expansion"]) if i >= BASE_CHANNELS else QtGui.QColor(COLORS["text"])
                p.setPen(QtGui.QPen(border_color, 2))
                p.drawEllipse(QtCore.QPointF(ex, ey), radius, radius)
                # Nome — Inter
                p.setFont(name_font)
                p.setPen(QtGui.QPen(QtGui.QColor("#000000")))
                p.drawText(QtCore.QRectF(ex - radius, ey - radius, 2*radius, 2*radius),
                           QtCore.Qt.AlignmentFlag.AlignCenter, name)
                # Valor numerico — JetBrains Mono
                p.setFont(value_font)
                p.setPen(QtGui.QPen(QtGui.QColor(COLORS["text_dim"])))
                p.drawText(QtCore.QRectF(ex - 40, ey + radius + 1, 80, 12),
                           QtCore.Qt.AlignmentFlag.AlignCenter, f"{self.raw_values[i]:.2f}")

        # Legenda
        bar_y = h - 26; bar_x0 = 30; bar_x1 = w - 30
        steps = 100
        for s in range(steps):
            t = s / (steps - 1)
            color = self._color_for_power(t)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QBrush(color))
            x0 = bar_x0 + s * (bar_x1 - bar_x0) / steps
            x1 = bar_x0 + (s + 1) * (bar_x1 - bar_x0) / steps
            p.drawRect(QtCore.QRectF(x0, bar_y, x1 - x0 + 1, 9))
        p.setPen(QtGui.QPen(QtGui.QColor(COLORS["text_dim"])))
        legend_font = QtGui.QFont(FONT_UI, 8, QtGui.QFont.Weight.Normal)
        p.setFont(legend_font)
        p.drawText(QtCore.QRectF(bar_x0, bar_y + 11, 60, 14),
                   QtCore.Qt.AlignmentFlag.AlignLeft, "baixo")
        p.drawText(QtCore.QRectF(bar_x1 - 60, bar_y + 11, 60, 14),
                   QtCore.Qt.AlignmentFlag.AlignRight, "alto")

    @staticmethod
    def _color_for_power(t):
        t = max(0.0, min(1.0, t))
        hue = int(270 * (1 - t))
        return QtGui.QColor.fromHsv(hue, 230, 255)


# ============================================================
# FocusMeterWidget
# ============================================================
class FocusMeterWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        title = QtWidgets.QLabel("Índice de Foco — β / (α + β)")
        title.setStyleSheet(f"color: {COLORS['accent']}; font-weight: bold; font-size: 11pt;")
        layout.addWidget(title)
        self.value_label = QtWidgets.QLabel("--")
        self.value_label.setStyleSheet(f"color: {COLORS['accent']}; font-size: 32pt; font-weight: bold;")
        self.value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.value_label)
        self.bar = QtWidgets.QProgressBar(); self.bar.setRange(0, 100)
        self.bar.setTextVisible(False); self.bar.setMaximumHeight(20)
        layout.addWidget(self.bar)
        self.state_label = QtWidgets.QLabel("aguardando...")
        self.state_label.setStyleSheet(f"color: {COLORS['text_dim']};")
        self.state_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.state_label)
        self.plot = pg.PlotWidget(enableMenu=False)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setYRange(0, 1); self.plot.setLabel("left", "Foco")
        self.plot.setMenuEnabled(False); self.plot.setMouseEnabled(x=False, y=False)
        self.curve = self.plot.plot(pen=pg.mkPen(COLORS["accent"], width=2))
        layout.addWidget(self.plot, stretch=1)
        self.history = deque([0.5] * 120, maxlen=120)

    def update_value(self, focus):
        self.history.append(focus)
        self.curve.setData(np.arange(len(self.history)), np.array(self.history))
        pct = int(focus * 100)
        self.value_label.setText(f"{focus:.2f}")
        self.bar.setValue(pct)
        if focus < 0.40:
            self.state_label.setText("RELAXADO / DISTRAIDO")
            self.state_label.setStyleSheet(f"color: {COLORS['warning']};")
        elif focus > 0.60:
            self.state_label.setText("FOCADO")
            self.state_label.setStyleSheet(f"color: {COLORS['accent']}; font-weight: bold;")
        else:
            self.state_label.setText("NEUTRO")
            self.state_label.setStyleSheet(f"color: {COLORS['text_dim']};")


# ============================================================
# EMGEnvelopeWidget
# ============================================================
class EMGEnvelopeWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        title = QtWidgets.QLabel("Envoltoria EMG (RMS suavizado)")
        title.setStyleSheet(f"color: {COLORS['accent']}; font-weight: bold; font-size: 11pt;")
        layout.addWidget(title)
        ch_row = QtWidgets.QHBoxLayout()
        ch_row.addWidget(QtWidgets.QLabel("Canal:"))
        self.channel_combo = QtWidgets.QComboBox()
        for i in range(MAX_CHANNELS):
            self.channel_combo.addItem(f"CH{i+1}")
        ch_row.addWidget(self.channel_combo); ch_row.addStretch()
        layout.addLayout(ch_row)
        self.plot = pg.PlotWidget(enableMenu=False)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel("left", "|sinal| (µV)")
        self.plot.setLabel("bottom", "Tempo", units="s")
        self.plot.setMenuEnabled(False)
        self.curve_raw = self.plot.plot(pen=pg.mkPen(COLORS["text_dim"], width=1))
        self.curve_env = self.plot.plot(pen=pg.mkPen(COLORS["accent"], width=2))
        layout.addWidget(self.plot, stretch=1)

    def set_num_channels(self, n):
        # Habilita/desabilita itens 9-16
        for i in range(MAX_CHANNELS):
            item = self.channel_combo.model().item(i)
            if item is not None:
                item.setEnabled(i < n)
        if self.channel_combo.currentIndex() >= n:
            self.channel_combo.setCurrentIndex(0)

    def update_signal(self, data, sample_rate=SAMPLE_RATE):
        if len(data) < 10:
            return
        t = np.arange(len(data)) / sample_rate
        env = SignalProcessor.compute_emg_envelope(data, sample_rate)
        self.curve_raw.setData(t, np.abs(data))
        self.curve_env.setData(t, env)


# ============================================================
# _EogGazeWidget — visualização de direção do olhar (EoG)
# ============================================================
class _EogGazeWidget(QtWidgets.QWidget):
    """Diagrama XY mostrando posição do olho em tempo real.

    Recebe coordenadas normalizadas em [-1, +1]. A origem (0,0) é o centro.
    +x = direita, +y = cima. Linhas-guia desenhadas a cada 0.5 unidade.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._gx = 0.0
        self._gy = 0.0
        self._trail = []  # últimas N posições para "rastro"
        self.setMinimumSize(180, 180)

    def set_gaze(self, gx, gy):
        # Clampa em [-1, +1]
        self._gx = max(-1.0, min(1.0, float(gx)))
        self._gy = max(-1.0, min(1.0, float(gy)))
        self._trail.append((self._gx, self._gy))
        if len(self._trail) > 30:
            self._trail.pop(0)
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w = self.width(); h = self.height()
        size = min(w, h)
        # Centro
        cx = w / 2; cy = h / 2
        r = size / 2 - 6
        # Fundo circular
        p.setBrush(QtGui.QBrush(QtGui.QColor(COLORS["surface_alt"])))
        p.setPen(QtGui.QPen(QtGui.QColor(COLORS["border"]), 1))
        p.drawEllipse(QtCore.QPointF(cx, cy), r, r)
        # Grade
        p.setPen(QtGui.QPen(QtGui.QColor(COLORS["border"]), 1,
                            QtCore.Qt.PenStyle.DashLine))
        p.drawLine(QtCore.QPointF(cx - r, cy), QtCore.QPointF(cx + r, cy))
        p.drawLine(QtCore.QPointF(cx, cy - r), QtCore.QPointF(cx, cy + r))
        for f in (0.33, 0.66):
            p.drawEllipse(QtCore.QPointF(cx, cy), r * f, r * f)
        # Rastro (alpha proporcional à idade)
        n = len(self._trail)
        for i, (gx, gy) in enumerate(self._trail):
            alpha = int(40 + 180 * i / max(1, n - 1))
            color = QtGui.QColor(SIGNAL_TYPE_COLORS["EoG"])
            color.setAlpha(alpha)
            p.setBrush(QtGui.QBrush(color))
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            px = cx + gx * r; py = cy - gy * r  # y invertido (Qt y+ é para baixo)
            p.drawEllipse(QtCore.QPointF(px, py), 3, 3)
        # Dot atual
        p.setBrush(QtGui.QBrush(QtGui.QColor(SIGNAL_TYPE_COLORS["EoG"])))
        p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.5))
        px = cx + self._gx * r; py = cy - self._gy * r
        p.drawEllipse(QtCore.QPointF(px, py), 7, 7)
        # Labels direcionais (triângulos Geometric Shapes — não-emoji)
        p.setPen(QtGui.QPen(QtGui.QColor(COLORS["text_dim"])))
        p.setFont(QtGui.QFont(FONT_UI, 9, QtGui.QFont.Weight.Bold))
        p.drawText(int(cx - 5), int(cy - r + 12), "▲")  # ▲
        p.drawText(int(cx - 5), int(cy + r - 2),  "▼")  # ▼
        p.drawText(int(cx + r - 14), int(cy + 5), "▶")  # ▶
        p.drawText(int(cx - r + 4),  int(cy + 5), "◀")  # ◀


# ============================================================
# _VirtualJoystickWidget — visualização do EMG Joystick
# ============================================================
class _VirtualJoystickWidget(QtWidgets.QWidget):
    """Widget que desenha um joystick virtual.

    Estado: x, y em [-1, +1]. Áreas N/S/L/O destacadas quando ativadas.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._x = 0.0; self._y = 0.0
        self._dead_zone = 0.15
        self.setMinimumSize(220, 220)

    def set_axes(self, x, y, dead_zone=0.15):
        self._x = max(-1.0, min(1.0, float(x)))
        self._y = max(-1.0, min(1.0, float(y)))
        self._dead_zone = dead_zone
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        w = self.width(); h = self.height()
        size = min(w, h)
        cx = w / 2; cy = h / 2
        r = size / 2 - 8
        # Quadrante de fundo
        p.setBrush(QtGui.QBrush(QtGui.QColor(COLORS["surface_alt"])))
        p.setPen(QtGui.QPen(QtGui.QColor(COLORS["border"]), 1.5))
        p.drawRect(QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r))
        # Cruz central
        p.setPen(QtGui.QPen(QtGui.QColor(COLORS["border"]), 1,
                            QtCore.Qt.PenStyle.DashLine))
        p.drawLine(QtCore.QPointF(cx - r, cy), QtCore.QPointF(cx + r, cy))
        p.drawLine(QtCore.QPointF(cx, cy - r), QtCore.QPointF(cx, cy + r))
        # Dead zone (círculo)
        p.setBrush(QtGui.QBrush(QtGui.QColor(COLORS["background"])))
        p.setPen(QtGui.QPen(QtGui.QColor(COLORS["text_dim"]), 1,
                            QtCore.Qt.PenStyle.DotLine))
        dz_r = r * self._dead_zone
        p.drawEllipse(QtCore.QPointF(cx, cy), dz_r, dz_r)
        # Direções (N/S/L/O) — destacadas se ativas
        # Triângulos Geometric Shapes (sem emoji)
        active_color = QtGui.QColor(SIGNAL_TYPE_COLORS["EMG"])
        inactive_color = QtGui.QColor(COLORS["text_dim"])
        p.setFont(QtGui.QFont(FONT_UI, 12, QtGui.QFont.Weight.Bold))
        # Up (y>0.3)
        c = active_color if self._y > 0.3 else inactive_color
        p.setPen(QtGui.QPen(c))
        p.drawText(int(cx - 6), int(cy - r + 18), "▲")
        # Down
        c = active_color if self._y < -0.3 else inactive_color
        p.setPen(QtGui.QPen(c))
        p.drawText(int(cx - 6), int(cy + r - 4), "▼")
        # Right
        c = active_color if self._x > 0.3 else inactive_color
        p.setPen(QtGui.QPen(c))
        p.drawText(int(cx + r - 16), int(cy + 6), "▶")
        # Left
        c = active_color if self._x < -0.3 else inactive_color
        p.setPen(QtGui.QPen(c))
        p.drawText(int(cx - r + 4), int(cy + 6), "◀")
        # Cursor (posição atual)
        p.setBrush(QtGui.QBrush(QtGui.QColor(SIGNAL_TYPE_COLORS["EMG"])))
        p.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.5))
        px = cx + self._x * r; py = cy - self._y * r
        p.drawEllipse(QtCore.QPointF(px, py), 10, 10)
        # Coordenadas (canto)
        p.setPen(QtGui.QPen(QtGui.QColor(COLORS["text"])))
        p.setFont(QtGui.QFont(FONT_DATA, 9))
        p.drawText(8, h - 6, f"X={self._x:+.2f}  Y={self._y:+.2f}")


# ============================================================
# _CommandPalette — busca global tipo VS Code (Ctrl+Shift+P)
# ============================================================
# Lista comandos + abas + voluntários + sessões. Usuário digita,
# filtro fuzzy, Enter executa.
class _CommandPalette(QtWidgets.QDialog):
    """Diálogo flutuante de busca global de comandos.

    Comandos são uma lista de dicts:
        {"label": "Abrir Conexão", "category": "Aba", "action": callable}
    """
    def __init__(self, commands, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pesquisar")
        self.setWindowFlags(QtCore.Qt.WindowType.Dialog |
                             QtCore.Qt.WindowType.FramelessWindowHint)
        self.setMinimumSize(540, 420)
        self.setStyleSheet(
            f"QDialog {{ background: {COLORS.get('surface', '#222')}; "
            f"border: 2px solid {COLORS.get('accent', '#a6e22e')}; "
            f"border-radius: 6px; }} "
            f"QLineEdit {{ background: {COLORS.get('surface_alt', '#333')}; "
            f"color: {COLORS.get('text', '#fff')}; border: 1px solid "
            f"{COLORS.get('border', '#555')}; padding: 8px; font-size: 13pt; }} "
            f"QListWidget {{ background: {COLORS.get('surface', '#222')}; "
            f"color: {COLORS.get('text', '#fff')}; border: none; "
            f"font-size: 11pt; }} "
            f"QListWidget::item {{ padding: 6px 8px; "
            f"border-bottom: 1px solid {COLORS.get('border', '#444')}; }} "
            f"QListWidget::item:selected {{ "
            f"background: {COLORS.get('accent', '#a6e22e')}; "
            f"color: {COLORS.get('background', '#000')}; }}"
        )
        self._commands = list(commands)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8); layout.setSpacing(6)
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Pesquisar abas, voluntários, sessões, comandos...")
        self.search_edit.textChanged.connect(self._refilter)
        layout.addWidget(self.search_edit)
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.itemActivated.connect(self._execute_item)
        self.list_widget.itemDoubleClicked.connect(self._execute_item)
        layout.addWidget(self.list_widget, stretch=1)
        # Atalhos de navegação
        QtGui.QShortcut(QtGui.QKeySequence("Esc"), self, activated=self.reject)
        QtGui.QShortcut(QtGui.QKeySequence("Down"), self,
                         activated=lambda: self._move_selection(+1))
        QtGui.QShortcut(QtGui.QKeySequence("Up"), self,
                         activated=lambda: self._move_selection(-1))
        QtGui.QShortcut(QtGui.QKeySequence("Return"), self,
                         activated=self._execute_current)
        self._refilter("")
        self.search_edit.setFocus()

    def _refilter(self, text):
        text = text.lower().strip()
        self.list_widget.clear()
        for cmd in self._commands:
            label = cmd.get("label", "")
            category = cmd.get("category", "")
            haystack = f"{label} {category}".lower()
            if text and not all(tok in haystack for tok in text.split()):
                continue
            it = QtWidgets.QListWidgetItem(f"[{category}]  {label}")
            it.setData(QtCore.Qt.ItemDataRole.UserRole, cmd)
            self.list_widget.addItem(it)
        if self.list_widget.count():
            self.list_widget.setCurrentRow(0)

    def _move_selection(self, delta):
        cur = self.list_widget.currentRow()
        new = (cur + delta) % max(1, self.list_widget.count())
        self.list_widget.setCurrentRow(new)

    def _execute_current(self):
        it = self.list_widget.currentItem()
        if it: self._execute_item(it)

    def _execute_item(self, item):
        cmd = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if cmd and callable(cmd.get("action")):
            try: cmd["action"]()
            except Exception as exc: print(f"[Palette] erro: {exc}")
        self.accept()


# ============================================================
# _SimulationOverlay — watermark diagonal "DADOS NÃO REAIS"
# ============================================================
# Quando o modo de aquisição é "Simulação" ou "Playback", este overlay é
# mostrado por cima do conteúdo central com uma marca d'água diagonal
# semi-transparente para evitar que demos sejam confundidos com sessões
# clínicas reais.
class _SimulationOverlay(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Não captura mouse — passa cliques para os widgets abaixo
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        # Fundo transparente
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._text = "SIMULAÇÃO - DADOS NÃO REAIS"
        self._color = QtGui.QColor(255, 100, 100, 35)  # vermelho semi-transparente

    def set_text(self, text):
        self._text = text
        self.update()

    def set_color(self, color):
        self._color = color
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        w = self.width(); h = self.height()
        # Texto repetido em diagonal — padrão discreto (espaçado) para não
        # competir com o sinal. Cor/alpha vêm de set_color (alpha baixo).
        font = QtGui.QFont("Inter", 24, QtGui.QFont.Weight.Bold)
        p.setFont(font)
        p.setPen(QtGui.QPen(self._color))
        # Rotaciona -30° e desenha algumas vezes, bem espaçado
        p.translate(w / 2, h / 2)
        p.rotate(-30)
        fm = QtGui.QFontMetrics(font)
        tw = fm.horizontalAdvance(self._text)
        th = fm.height()
        # Tile o texto num padrão esparso (menos repetições)
        for y in range(-int(h*1.5), int(h*1.5), th * 6):
            for x in range(-int(w*1.5), int(w*1.5), tw + 220):
                p.drawText(x, y, self._text)


# ============================================================
# BluetoothScanThread — varredura BLE async em thread separada
# ============================================================
# Usa bleak (BLE) se disponível. Faz scan de 8s e emite a lista de
# devices encontrados via Qt signal. Não bloqueia a UI principal.
#
# Em Windows, bleak usa as APIs nativas WinRT (sem precisar de drivers extras).
# Para dispositivos BT clássicos (HC-05, módulos seriais), o usuário deve
# parear pelo Windows e a porta COM virtual aparecerá no combo de portas.
# ============================================================
try:
    import bleak as _bleak_module
    HAS_BLEAK = True
except Exception:
    HAS_BLEAK = False


class _BluetoothScanThread(QtCore.QThread):
    """Thread que executa scan BLE async (8 s) e emite devices ao concluir.

    Sinais:
        scan_progress(int seconds_elapsed)
        scan_done(list of dicts {name, address, rssi, type})
        scan_failed(str error_message)
    """
    scan_progress = QtCore.Signal(int)
    scan_done = QtCore.Signal(list)
    scan_failed = QtCore.Signal(str)

    def __init__(self, duration_s=8.0, parent=None):
        super().__init__(parent)
        self._duration = duration_s

    def run(self):
        if not HAS_BLEAK:
            self.scan_failed.emit(
                "Biblioteca 'bleak' não instalada. "
                "Instale com: pip install bleak"
            )
            return
        import asyncio
        try:
            from bleak import BleakScanner
        except Exception as exc:
            self.scan_failed.emit(f"Erro ao importar bleak: {exc}")
            return

        async def _do_scan():
            devices = []
            try:
                # discover() retorna dict[address, (BLEDevice, AdvertisementData)]
                # em bleak >= 0.20. Para versões antigas, usa scanner manual.
                if hasattr(BleakScanner, "discover"):
                    found = await BleakScanner.discover(
                        timeout=self._duration, return_adv=True)
                    for addr, (dev, adv) in found.items():
                        devices.append({
                            "name":    dev.name or "(sem nome)",
                            "address": dev.address,
                            "rssi":    getattr(adv, "rssi", None),
                            "type":    "BLE",
                        })
                else:
                    found = await BleakScanner.discover(timeout=self._duration)
                    for d in found:
                        devices.append({
                            "name":    d.name or "(sem nome)",
                            "address": d.address,
                            "rssi":    getattr(d, "rssi", None),
                            "type":    "BLE",
                        })
            except Exception as exc:
                raise exc
            return devices

        try:
            # Cria event loop dedicado para esta thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            devices = loop.run_until_complete(_do_scan())
            loop.close()
            self.scan_done.emit(devices)
        except Exception as exc:
            self.scan_failed.emit(f"Falha no scan BLE: {exc}")


# ============================================================
# LauncherScreen — Tela de boas-vindas / dashboard inicial
# ============================================================
# Aparece ANTES da janela principal e direciona o fluxo de trabalho do
# usuário. Reduz sobrecarga cognitiva: em vez de cair direto em 4 grupos
# × 5+ sub-abas, o usuário escolhe a intenção (Nova Coleta / Analisar /
# BCI / Simulação) + configura hardware básico antes do app abrir.
#
# Retorna um dict de "choice" com:
#   {"mode": "live"|"offline"|"bci"|"sim",
#    "port": "COM3" or None,
#    "expansion_16ch": bool,
#    "acquisition_type": "EEG"|"EMG"|"ECG"|"Hibrido",
#    "volunteer_dir": str or None}
# ============================================================
class _WorkflowCard(QtWidgets.QFrame):
    """Card grande clicável para a tela inicial.

    Recebe as cores via parâmetro (tema). Hover é controlado por property
    [hover="true"] aplicada via QSS no nível do diálogo.
    """
    clicked = QtCore.Signal()

    def __init__(self, title, subtitle, icon_text, accent_color,
                 text_color="#1a1a1a", dim_color="#666666", parent=None):
        super().__init__(parent)
        self.setObjectName("workflowCard")
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        # Largura mínima enxuta para que 2 cards caibam no painel central mesmo
        # na janela mínima (1200px); crescem via size policy Expanding.
        self.setMinimumSize(188, 168)
        self.setProperty("hover", False)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(6)
        # Ícone (símbolo tipográfico — sem emoji, segue tema)
        self.icon_lbl = QtWidgets.QLabel(icon_text)
        self.icon_lbl.setObjectName("cardIcon")
        self.icon_lbl.setStyleSheet(
            f"color: {accent_color}; font-size: 36pt; font-weight: bold; "
            f"background: transparent;")
        self.icon_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.icon_lbl)
        # Título
        self.title_lbl = QtWidgets.QLabel(title)
        self.title_lbl.setObjectName("cardTitle")
        self.title_lbl.setStyleSheet(
            f"color: {text_color}; font-size: 13pt; font-weight: bold; "
            f"background: transparent;")
        self.title_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.title_lbl)
        # Subtítulo
        self.sub_lbl = QtWidgets.QLabel(subtitle)
        self.sub_lbl.setObjectName("cardSubtitle")
        self.sub_lbl.setStyleSheet(
            f"color: {dim_color}; font-size: 9pt; background: transparent;")
        self.sub_lbl.setWordWrap(True)
        self.sub_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.sub_lbl)

        self._accent = accent_color

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)

    def enterEvent(self, ev):
        self.setProperty("hover", True)
        self.style().unpolish(self); self.style().polish(self)
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self.setProperty("hover", False)
        self.style().unpolish(self); self.style().polish(self)
        super().leaveEvent(ev)


class LauncherScreen(QtWidgets.QDialog):
    """Tela inicial. Bloqueia até o usuário fazer uma escolha (ou sair).

    Uso:
        launcher = LauncherScreen(config, volunteers_mgr)
        if launcher.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            choice = launcher.get_choice()
            # ...

    Cores: segue o tema global (COLORS dict) — não tem paleta fixa.
    """

    # Sinal opcional para integrações externas
    launch_requested = QtCore.Signal(dict)

    @staticmethod
    def _shift_color(hex_color, delta):
        """Clareia/escurece uma cor hex. delta + clareia, - escurece. [-100..+100]."""
        try:
            c = QtGui.QColor(hex_color)
            h, s, v, a = c.getHsv()
            v = max(0, min(255, v + delta))
            c.setHsv(h, s, v, a)
            return c.name()
        except Exception:
            return hex_color

    @staticmethod
    def _shrink_combo(combo, min_chars=8):
        """Evita que itens de texto longo (porta COM, nome de voluntário)
        forcem a largura do painel lateral — o que espremia o painel central
        e cortava os cards. O combo passa a dimensionar pelo mínimo de
        caracteres e a encolher/elidir; o popup mantém a largura dos itens."""
        try:
            combo.setSizeAdjustPolicy(
                QtWidgets.QComboBox.SizeAdjustPolicy
                .AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(min_chars)
            combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                QtWidgets.QSizePolicy.Policy.Fixed)
            combo.setMinimumWidth(0)
        except Exception:
            pass

    def _theme(self):
        """Atalho para a paleta do tema atual + cores derivadas."""
        return {
            "bg":         COLORS.get("background",  "#1e1e1e"),
            "panel":      COLORS.get("surface",     "#2d2d2d"),
            "panel_alt":  COLORS.get("surface_alt", "#363636"),
            "border":     COLORS.get("border",      "#404040"),
            "text":       COLORS.get("text",        "#ffffff"),
            "text_dim":   COLORS.get("text_dim",    "#aaaaaa"),
            "accent":     COLORS.get("accent",      "#a6e22e"),
            "accent_dim": COLORS.get("accent_dim",  "#7eb820"),
            "danger":     COLORS.get("error",       "#ee5566"),
            "warning":    COLORS.get("warning",     "#eebb33"),
            # Cores derivadas
            "hover":      self._shift_color(COLORS.get("surface_alt", "#363636"), +10),
            "card_bg":    COLORS.get("surface_alt", "#363636"),
        }

    def __init__(self, config=None, volunteers_mgr=None, parent=None):
        super().__init__(parent)
        self.config = config
        self.volunteers = volunteers_mgr
        self.choice = None   # dict preenchido ao clicar em um card
        self._open_settings_request = False  # se o usuário clicou no engrenagem

        self.setWindowTitle(f"{APP_NAME} — Launcher")
        self.setMinimumSize(1200, 680)
        self.resize(1280, 720)
        # Ícone (se existir) — resolve em Documents/SCRIPT_DIR/_MEIPASS (.exe)
        ico_path = _resolve_asset("app_brain.ico", os.path.join(SCRIPT_DIR, "app_brain.ico"))
        if os.path.exists(ico_path):
            self.setWindowIcon(QtGui.QIcon(ico_path))

        self.setStyleSheet(self._build_stylesheet())
        self._build_ui()
        self._populate_data()

    # ----- Choice API -----
    def get_choice(self):
        """Retorna o dict de escolha (preenchido quando o usuário clica em um card)."""
        return self.choice

    def settings_requested(self):
        """True se o usuário pediu para abrir as Configurações ao invés de um modo."""
        return self._open_settings_request

    # ----- UI -----
    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)

        # ----- Body: 3 painéis lado a lado -----
        body = QtWidgets.QWidget()
        body.setObjectName("body")
        body_layout = QtWidgets.QHBoxLayout(body)
        body_layout.setContentsMargins(16, 16, 16, 8); body_layout.setSpacing(12)
        # Proporção 3:6:3 — painel central recebe mais espaço para evitar
        # cortar o título "O que você deseja fazer hoje?"
        body_layout.addWidget(self._build_left_panel(),  stretch=3)
        body_layout.addWidget(self._build_center_panel(), stretch=6)
        body_layout.addWidget(self._build_right_panel(), stretch=3)
        root.addWidget(body, stretch=1)

        # ----- Footer -----
        root.addWidget(self._build_footer())

    # ----- Painel esquerdo -----
    def _build_left_panel(self):
        th = self._theme()
        panel = QtWidgets.QFrame()
        panel.setObjectName("sidePanel")
        # Largura limitada: impede que os painéis laterais "comam" o painel
        # central (o que cortava os cards de Analisar Dados / Modo Simulação).
        panel.setMaximumWidth(340)
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(16, 16, 16, 16); v.setSpacing(10)

        # Logo no topo
        logo_box = QtWidgets.QFrame()
        logo_box.setObjectName("logoBox")
        lb = QtWidgets.QVBoxLayout(logo_box)
        lb.setContentsMargins(12, 12, 12, 12)
        lb.setSpacing(4)
        # Brand mark (placeholder se PNG não estiver disponível)
        brand_lbl = QtWidgets.QLabel("◢")
        brand_lbl.setStyleSheet(
            f"color: {th['accent']}; font-size: 42pt; font-weight: bold; "
            f"background: transparent; letter-spacing: 0px;")
        brand_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lb.addWidget(brand_lbl)
        title_lbl = QtWidgets.QLabel(APP_NAME)
        title_lbl.setStyleSheet(
            f"color: {th['text']}; font-size: 13pt; font-weight: bold; "
            f"background: transparent;")
        title_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lb.addWidget(title_lbl)
        sub_lbl = QtWidgets.QLabel(f"{APP_EDITION} · v{APP_VERSION}")
        sub_lbl.setStyleSheet(
            f"color: {th['text_dim']}; font-size: 9pt; background: transparent;")
        sub_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lb.addWidget(sub_lbl)
        v.addWidget(logo_box)

        # ---- Voluntário ativo ----
        v.addWidget(self._section_label(tr("Voluntário Ativo")))
        vol_row = QtWidgets.QHBoxLayout()
        vol_row.setSpacing(6)
        self.volunteer_combo = QtWidgets.QComboBox()
        self.volunteer_combo.setObjectName("volunteerCombo")
        self._shrink_combo(self.volunteer_combo)
        vol_row.addWidget(self.volunteer_combo, stretch=1)
        self.new_volunteer_btn = QtWidgets.QPushButton(tr("+ Novo"))
        self.new_volunteer_btn.setObjectName("smallBtn")
        self.new_volunteer_btn.setMaximumWidth(90)
        self.new_volunteer_btn.clicked.connect(self._on_new_volunteer_clicked)
        vol_row.addWidget(self.new_volunteer_btn)
        v.addLayout(vol_row)

        # ---- Sessões recentes ----
        v.addWidget(self._section_label(tr("Sessões Recentes")))
        self.recent_list = QtWidgets.QListWidget()
        self.recent_list.setObjectName("recentList")
        self.recent_list.itemDoubleClicked.connect(self._on_recent_session_open)
        v.addWidget(self.recent_list, stretch=1)
        # Botão para abrir CSV manualmente
        open_csv_btn = QtWidgets.QPushButton(tr("Abrir CSV manualmente..."))
        open_csv_btn.setObjectName("ghostBtn")
        open_csv_btn.clicked.connect(self._on_open_csv_manual)
        v.addWidget(open_csv_btn)

        return panel

    # ----- Painel central -----
    def _build_center_panel(self):
        th = self._theme()
        panel = QtWidgets.QFrame()
        panel.setObjectName("centerPanel")
        v = QtWidgets.QVBoxLayout(panel)
        # Margens reduzidas + sem stretches estranhos para não comprimir os cards
        v.setContentsMargins(20, 24, 20, 20); v.setSpacing(10)

        # Título grande — auto-shrink se a janela for menor que o ideal.
        # Diminuímos a fonte e usamos word-wrap para evitar corte.
        title = QtWidgets.QLabel(tr("O que você deseja fazer hoje?"))
        title.setObjectName("centerTitle")
        title.setStyleSheet(
            f"color: {th['text']}; font-size: 18pt; font-weight: bold; "
            f"background: transparent; letter-spacing: 0.5px; padding: 0;")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setWordWrap(True)
        v.addWidget(title)

        subtitle = QtWidgets.QLabel(tr("Escolha um fluxo de trabalho para começar"))
        subtitle.setStyleSheet(
            f"color: {th['text_dim']}; font-size: 11pt; "
            f"background: transparent; padding-bottom: 6px;")
        subtitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        v.addWidget(subtitle)

        # Grid 2x2 de cards. Cada card EXPANDE para preencher sua coluna e o
        # bloco é limitado em largura + centralizado por espaçadores laterais —
        # assim os cards NUNCA estouram o painel central (o que cortava o
        # subtítulo de "Analisar Dados" e "Modo Simulação" na coluna direita).
        cards_wrapper = QtWidgets.QWidget()
        cards = QtWidgets.QGridLayout(cards_wrapper)
        cards.setHorizontalSpacing(16); cards.setVerticalSpacing(16)
        cards.setContentsMargins(0, 0, 0, 0)

        # Os 4 fluxos. (mode, título, subtítulo, ícone-texto, cor de destaque)
        # Ícones: caracteres tipográficos Unicode (Geometric Shapes block) —
        # SEM emojis. Renderizados como texto puro pelo Qt.
        flows = [
            ("live",    tr("Nova Coleta"),
             tr("Conectar ao hardware e gravar uma sessão em tempo real"),
             "▶", th["accent"]),
            ("offline", tr("Analisar Dados"),
             tr("Abrir um CSV e explorar análises offline (FFT, ERP, ERS/ERD)"),
             "■", th["accent"]),
            ("bci",     tr("Aplicações BCI"),
             tr("Biofeedback interativo: Focus, EMG Joystick, SSVEP"),
             "◆", th["warning"]),
            ("sim",     tr("Modo Simulação"),
             tr("Gerar dados sintéticos para testar a interface sem hardware"),
             "◯", th["text_dim"]),
        ]
        self._cards = {}
        for i, (mode, t, s, ic, col) in enumerate(flows):
            r, c = i // 2, i % 2
            card = _WorkflowCard(t, s, ic, col,
                                  text_color=th["text"],
                                  dim_color=th["text_dim"])
            card.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                               QtWidgets.QSizePolicy.Policy.Expanding)
            card.clicked.connect(lambda m=mode: self._on_card_clicked(m))
            cards.addWidget(card, r, c)
            self._cards[mode] = card
        # Faz as 2 colunas e linhas terem o mesmo stretch (cards iguais)
        cards.setColumnStretch(0, 1); cards.setColumnStretch(1, 1)
        cards.setRowStretch(0, 1); cards.setRowStretch(1, 1)
        # O bloco preenche a largura do painel central. Como os painéis
        # laterais têm largura máxima limitada, o central sempre recebe espaço
        # suficiente para os 2 cards lado a lado, sem corte — em qualquer
        # tamanho de janela (>= 1200 px). Os cards crescem juntos (Expanding).
        v.addWidget(cards_wrapper, stretch=1)

        return panel

    # ----- Painel direito -----
    def _build_right_panel(self):
        th = self._theme()
        panel = QtWidgets.QFrame()
        panel.setObjectName("sidePanel")
        panel.setMaximumWidth(340)  # idem painel esquerdo (não espremer o centro)
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(16, 16, 16, 16); v.setSpacing(10)

        v.addWidget(self._section_label(tr("Pré-Flight Check")))
        hint = QtWidgets.QLabel(tr(
            "Configure o hardware antes de iniciar a coleta. "
            "Estes parâmetros são aplicados ao iniciar."))
        hint.setStyleSheet(
            f"color: {th['text_dim']}; font-size: 9pt; background: transparent;")
        hint.setWordWrap(True)
        v.addWidget(hint)

        # ---- Bloco: Hardware ----
        hw_box = QtWidgets.QFrame()
        hw_box.setObjectName("subPanel")
        hl = QtWidgets.QVBoxLayout(hw_box)
        hl.setContentsMargins(12, 10, 12, 12); hl.setSpacing(8)
        hl.addWidget(self._sub_label(tr("Configuração de Hardware")))
        # Porta COM
        port_row = QtWidgets.QHBoxLayout(); port_row.setSpacing(6)
        port_row.addWidget(QtWidgets.QLabel(tr("Porta:")))
        self.launcher_port_combo = QtWidgets.QComboBox()
        # Não deixar o texto longo da porta (ex.: "COM11 — Serial Padrão por
        # link Bluetooth") forçar a largura do painel: o combo encolhe e elide,
        # abrindo espaço para o painel central (evita corte dos cards).
        self._shrink_combo(self.launcher_port_combo)
        port_row.addWidget(self.launcher_port_combo, stretch=1)
        self.refresh_ports_btn = QtWidgets.QPushButton(tr("Atualizar"))
        self.refresh_ports_btn.setObjectName("smallBtn")
        self.refresh_ports_btn.setMaximumWidth(90)
        self.refresh_ports_btn.setToolTip(tr("Atualizar lista de portas COM"))
        self.refresh_ports_btn.clicked.connect(self._refresh_ports)
        port_row.addWidget(self.refresh_ports_btn)
        hl.addLayout(port_row)
        # Seletor de número de canais (8 -> 64, em passos de 8). Substitui o
        # antigo checkbox binário "16 canais": agora o usuário escolhe a
        # quantidade exata ANTES de entrar no programa.
        ch_row = QtWidgets.QHBoxLayout(); ch_row.setSpacing(6)
        ch_row.addWidget(QtWidgets.QLabel(tr("Número de canais:")))
        self.expansion_combo = QtWidgets.QComboBox()
        self.expansion_combo.setToolTip(tr(
            "Placa base = 8 canais. Cada módulo de expansão adiciona +8 "
            "(até 64). Aplicado ao entrar no programa."))
        for n in EXPANSION_STEPS:
            if n == BASE_CHANNELS:
                self.expansion_combo.addItem(tr("8 canais (placa base)"), n)
            else:
                mods = (n - BASE_CHANNELS) // BASE_CHANNELS
                self.expansion_combo.addItem(
                    f"{n} " + tr("canais") + f"  (+{mods} " +
                    (tr("módulo") if mods == 1 else tr("módulos")) + ")", n)
        self.expansion_combo.setCurrentIndex(0)  # default: 8
        self._shrink_combo(self.expansion_combo)
        ch_row.addWidget(self.expansion_combo, stretch=1)
        hl.addLayout(ch_row)
        v.addWidget(hw_box)

        # ---- Bloco: Tipo de Aquisição ----
        type_box = QtWidgets.QFrame()
        type_box.setObjectName("subPanel")
        tl = QtWidgets.QVBoxLayout(type_box)
        tl.setContentsMargins(12, 10, 12, 12); tl.setSpacing(6)
        tl.addWidget(self._sub_label(tr("Tipo de Aquisição")))
        self.acquisition_group = QtWidgets.QButtonGroup(self)
        self.acquisition_group.setExclusive(True)
        self._acq_radios = {}
        for key, label, tip in (
            ("EEG", tr("Apenas EEG"),
             tr("Todos os canais marcados como EEG. Filtro 0.5-70 Hz + notch.")),
            ("EMG", tr("Apenas EMG"),
             tr("Todos os canais marcados como EMG. Filtro 20-Nyquist + notch.")),
            ("ECG", tr("Apenas ECG"),
             tr("Todos os canais marcados como ECG. Filtro 0.5-100 Hz + notch.")),
            ("Hibrido", tr("Híbrido (multimodal)"),
             tr("Mistura: 1-8 EEG / 9-12 EMG / 13-14 ECG / 15-16 EoG.")),
        ):
            rb = QtWidgets.QRadioButton(label)
            rb.setObjectName("acqRadio")
            rb.setToolTip(tip)
            tl.addWidget(rb)
            self.acquisition_group.addButton(rb)
            self._acq_radios[key] = rb
        # Default: EEG
        self._acq_radios["EEG"].setChecked(True)
        v.addWidget(type_box)

        # Espaço
        v.addStretch()

        # ---- Status atual (resumo) ----
        self.summary_lbl = QtWidgets.QLabel("")
        self.summary_lbl.setObjectName("summaryLabel")
        self.summary_lbl.setWordWrap(True)
        self.summary_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self.summary_lbl)
        self._refresh_summary()
        # Re-renderiza summary ao mudar opções
        self.launcher_port_combo.currentIndexChanged.connect(self._refresh_summary)
        self.expansion_combo.currentIndexChanged.connect(self._refresh_summary)
        for rb in self._acq_radios.values():
            rb.toggled.connect(self._refresh_summary)

        return panel

    def _selected_channels(self):
        """Número de canais escolhido no Pré-Flight (8..64)."""
        if hasattr(self, "expansion_combo"):
            val = self.expansion_combo.currentData()
            if isinstance(val, int):
                return val
        return BASE_CHANNELS

    # ----- Rodapé -----
    def _build_footer(self):
        th = self._theme()
        foot = QtWidgets.QFrame()
        foot.setObjectName("footer")
        h = QtWidgets.QHBoxLayout(foot)
        h.setContentsMargins(16, 8, 16, 12); h.setSpacing(8)

        self.settings_btn = QtWidgets.QPushButton(tr("Configurações"))
        self.settings_btn.setObjectName("ghostBtn")
        self.settings_btn.setToolTip(tr("Abrir o aplicativo direto na aba Configurações"))
        self.settings_btn.clicked.connect(self._on_settings_clicked)
        h.addWidget(self.settings_btn)

        h.addStretch()

        info_lbl = QtWidgets.QLabel(f"© {APP_YEAR} {APP_AUTHORS}")
        info_lbl.setStyleSheet(
            f"color: {th['text_dim']}; font-size: 9pt; background: transparent;")
        h.addWidget(info_lbl)

        h.addStretch()

        self.exit_btn = QtWidgets.QPushButton(tr("Sair"))
        self.exit_btn.setObjectName("dangerBtn")
        self.exit_btn.clicked.connect(self.reject)
        h.addWidget(self.exit_btn)

        return foot

    # ----- Helpers de UI -----
    def _section_label(self, text):
        th = self._theme()
        lbl = QtWidgets.QLabel(text.upper())
        lbl.setObjectName("sectionLabel")
        lbl.setStyleSheet(
            f"color: {th['accent']}; font-size: 10pt; font-weight: bold; "
            f"background: transparent; letter-spacing: 1.5px; padding-top: 4px;")
        return lbl

    def _sub_label(self, text):
        th = self._theme()
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet(
            f"color: {th['text']}; font-size: 10pt; font-weight: bold; "
            f"background: transparent;")
        return lbl

    # ----- Populadores -----
    def _populate_data(self):
        self._populate_volunteers()
        self._populate_recent_sessions()
        self._refresh_ports()

    def _populate_volunteers(self):
        self.volunteer_combo.clear()
        self.volunteer_combo.addItem("(nenhum — selecione ou cadastre)", None)
        if self.volunteers is None:
            return
        try:
            vols = self.volunteers.list_volunteers()
        except Exception as exc:
            print(f"[Launcher] erro listando voluntários: {exc}")
            return
        for prof in vols:
            vid = prof.get("vid", "?")
            nome = prof.get("nome", "")
            label = f"{vid} — {nome}" if nome else vid
            self.volunteer_combo.addItem(label, prof.get("_dirname"))

    def _populate_recent_sessions(self):
        self.recent_list.clear()
        if self.config is None:
            return
        base = getattr(self.config, "save_directory", None)
        if not base or not os.path.isdir(base):
            self.recent_list.addItem("(diretório de sessões vazio)")
            return
        # Coleta CSVs em subpastas (até 50)
        found = []
        try:
            # 1) Sessões soltas
            for entry in os.listdir(base):
                full = os.path.join(base, entry)
                if os.path.isdir(full):
                    csv = os.path.join(full, "data.csv")
                    if os.path.isfile(csv):
                        found.append((os.path.getmtime(csv), entry, csv))
            # 2) Sessões dentro de volunteers/<VID_Name>/
            vol_root = os.path.join(base, "volunteers")
            if os.path.isdir(vol_root):
                for vol_dir in os.listdir(vol_root):
                    full_v = os.path.join(vol_root, vol_dir)
                    if os.path.isdir(full_v):
                        for sess in os.listdir(full_v):
                            full_s = os.path.join(full_v, sess)
                            csv = os.path.join(full_s, "data.csv")
                            if os.path.isfile(csv):
                                found.append((os.path.getmtime(csv),
                                              f"{vol_dir} / {sess}", csv))
        except Exception as exc:
            print(f"[Launcher] erro listando sessões: {exc}")
        # Ordena por mtime desc e mostra até 30
        found.sort(reverse=True)
        if not found:
            it = QtWidgets.QListWidgetItem("(nenhuma sessão gravada ainda)")
            try:
                it.setForeground(QtGui.QColor(self._theme()["text_dim"]))
            except Exception:
                pass
            it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            self.recent_list.addItem(it)
            return
        for mtime, name, full_csv in found[:30]:
            import datetime as _dt
            ts = _dt.datetime.fromtimestamp(mtime).strftime("%d/%m %H:%M")
            label = f"  {name}\n  └─ {ts}"
            it = QtWidgets.QListWidgetItem(label)
            it.setData(QtCore.Qt.ItemDataRole.UserRole, full_csv)
            self.recent_list.addItem(it)

    def _refresh_ports(self):
        cur_text = self.launcher_port_combo.currentText() if hasattr(self, "launcher_port_combo") else ""
        self.launcher_port_combo.clear()
        try:
            import serial.tools.list_ports as _lp
            ports = list(_lp.comports())
        except Exception:
            ports = []
        if not ports:
            self.launcher_port_combo.addItem("(nenhuma porta detectada)", None)
        else:
            for p in ports:
                self.launcher_port_combo.addItem(
                    f"{p.device} — {p.description}", p.device)
        # Tenta restaurar seleção
        idx = self.launcher_port_combo.findText(cur_text)
        if idx >= 0:
            self.launcher_port_combo.setCurrentIndex(idx)

    def _refresh_summary(self):
        # Resumo bonito no canto direito inferior
        if not hasattr(self, "summary_lbl"):
            return
        th = self._theme()
        port = self.launcher_port_combo.currentData() if hasattr(self, "launcher_port_combo") else None
        ch = f"{self._selected_channels()} canais"
        acq = next((k for k, rb in self._acq_radios.items() if rb.isChecked()), "EEG")
        text = (f"<b>{tr('Setup pronto:')}</b><br>"
                f"{tr('Porta:')} <span style='color:{th['accent']}'>{port or '—'}</span> · "
                f"<span style='color:{th['accent']}'>{ch}</span> · "
                f"{tr('Modo:')} <span style='color:{th['accent']}'>{acq}</span>")
        self.summary_lbl.setText(text)
        self.summary_lbl.setStyleSheet(
            f"color: {th['text_dim']}; font-size: 9pt; "
            f"background: {th['panel_alt']}; "
            f"border: 1px solid {th['border']}; "
            f"border-radius: 4px; padding: 8px;")

    # ----- Event handlers -----
    def _on_card_clicked(self, mode):
        """Usuário escolheu um fluxo — fecha o dialog com a escolha."""
        port = self.launcher_port_combo.currentData() if hasattr(self, "launcher_port_combo") else None
        acq  = next((k for k, rb in self._acq_radios.items() if rb.isChecked()), "EEG")
        vol_dir = self.volunteer_combo.currentData() if hasattr(self, "volunteer_combo") else None
        n_ch = self._selected_channels()
        self.choice = {
            "mode": mode,
            "port": port,
            "num_channels": n_ch,
            "expansion_16ch": (n_ch > BASE_CHANNELS),
            "acquisition_type": acq,
            "volunteer_dir": vol_dir,
            "selected_csv": None,
        }
        # Sem backend complexo — apenas imprime para confirmar
        print(f"[Launcher] Card clicado: mode={mode!r}, port={port!r}, "
              f"ch={n_ch}, "
              f"acq={acq!r}, vol_dir={vol_dir!r}")
        self.launch_requested.emit(self.choice)
        self.accept()

    def _on_new_volunteer_clicked(self):
        print("[Launcher] Botão clicado: + Novo (voluntário)")
        # O cadastro completo é feito na janela principal — apenas sinalizamos
        # via choice especial. Mas como ainda não escolheu modo, oferecemos:
        QtWidgets.QMessageBox.information(
            self, "Cadastrar voluntário",
            "O cadastro completo será aberto após você iniciar um fluxo.\n"
            "Você pode cadastrar na aba <b>Voluntários</b> dentro do aplicativo.")

    def _on_recent_session_open(self, item):
        csv_path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not csv_path: return
        print(f"[Launcher] Sessão recente: {csv_path}")
        n_ch = self._selected_channels()
        self.choice = {
            "mode": "offline",
            "port": None,
            "num_channels": n_ch,
            "expansion_16ch": (n_ch > BASE_CHANNELS),
            "acquisition_type": "EEG",
            "volunteer_dir": None,
            "selected_csv": csv_path,
        }
        self.launch_requested.emit(self.choice)
        self.accept()

    def _on_open_csv_manual(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Abrir CSV de sessão",
            getattr(self.config, "save_directory", os.path.expanduser("~")),
            "CSV (*.csv);;Todos (*)")
        if not path: return
        print(f"[Launcher] CSV manual: {path}")
        n_ch = self._selected_channels()
        self.choice = {
            "mode": "offline",
            "port": None,
            "num_channels": n_ch,
            "expansion_16ch": (n_ch > BASE_CHANNELS),
            "acquisition_type": "EEG",
            "volunteer_dir": None,
            "selected_csv": path,
        }
        self.launch_requested.emit(self.choice)
        self.accept()

    def _on_settings_clicked(self):
        print("[Launcher] Botão clicado: Configurações")
        self._open_settings_request = True
        n_ch = self._selected_channels()
        self.choice = {
            "mode": "settings",
            "port": None,
            "num_channels": n_ch,
            "expansion_16ch": (n_ch > BASE_CHANNELS),
            "acquisition_type": "EEG",
            "volunteer_dir": None,
            "selected_csv": None,
        }
        self.launch_requested.emit(self.choice)
        self.accept()

    # ----- Stylesheet -----
    def _build_stylesheet(self):
        th = self._theme()
        c = {
            "BG":         th["bg"],
            "PANEL":      th["panel"],
            "PANEL_ALT":  th["panel_alt"],
            "BORDER":     th["border"],
            "TEXT":       th["text"],
            "TEXT_DIM":   th["text_dim"],
            "ACCENT":     th["accent"],
            "ACCENT_DIM": th["accent_dim"],
            "HOVER":      th["hover"],
            "DANGER":     th["danger"],
        }
        return f"""
        QDialog {{
            background-color: {c['BG']};
            color: {c['TEXT']};
        }}
        QWidget {{
            font-family: "Segoe UI", "Inter", sans-serif;
            color: {c['TEXT']};
        }}

        /* ---- Painéis ---- */
        QFrame#sidePanel, QFrame#centerPanel {{
            background-color: {c['PANEL']};
            border: 1px solid {c['BORDER']};
            border-radius: 8px;
        }}
        QFrame#subPanel, QFrame#logoBox {{
            background-color: {c['PANEL_ALT']};
            border: 1px solid {c['BORDER']};
            border-radius: 6px;
        }}

        /* ---- Workflow Cards ---- */
        QFrame#workflowCard {{
            background-color: {c['PANEL_ALT']};
            border: 2px solid {c['BORDER']};
            border-radius: 10px;
        }}
        QFrame#workflowCard[hover="true"] {{
            background-color: {c['HOVER']};
            border: 2px solid {c['ACCENT']};
        }}

        /* ---- Labels ---- */
        QLabel {{
            background: transparent;
        }}

        /* ---- Combos / Lists ---- */
        QComboBox, QListWidget {{
            background-color: {c['PANEL_ALT']};
            border: 1px solid {c['BORDER']};
            border-radius: 4px;
            padding: 6px 8px;
            color: {c['TEXT']};
            selection-background-color: {c['ACCENT_DIM']};
            selection-color: {c['BG']};
        }}
        QComboBox:hover {{
            border: 1px solid {c['ACCENT']};
        }}
        QComboBox::drop-down {{
            border: none; width: 24px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {c['PANEL_ALT']};
            border: 1px solid {c['BORDER']};
            color: {c['TEXT']};
            selection-background-color: {c['ACCENT']};
            selection-color: {c['BG']};
        }}
        QListWidget::item {{
            padding: 6px 4px;
            border-bottom: 1px solid {c['BORDER']};
        }}
        QListWidget::item:hover {{
            background-color: {c['HOVER']};
        }}
        QListWidget::item:selected {{
            background-color: {c['ACCENT_DIM']};
            color: {c['BG']};
        }}

        /* ---- Botões ---- */
        QPushButton {{
            background-color: {c['ACCENT']};
            color: {c['BG']};
            border: 1px solid {c['ACCENT_DIM']};
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {c['ACCENT_DIM']};
        }}
        QPushButton:pressed {{
            background-color: {c['ACCENT_DIM']};
            border: 1px solid {c['ACCENT']};
        }}

        QPushButton#smallBtn {{
            background-color: {c['PANEL_ALT']};
            color: {c['ACCENT']};
            border: 1px solid {c['BORDER']};
            padding: 6px 10px;
        }}
        QPushButton#smallBtn:hover {{
            background-color: {c['HOVER']};
            border: 1px solid {c['ACCENT']};
        }}

        QPushButton#ghostBtn {{
            background-color: transparent;
            color: {c['TEXT_DIM']};
            border: 1px solid {c['BORDER']};
            padding: 6px 12px;
        }}
        QPushButton#ghostBtn:hover {{
            color: {c['ACCENT']};
            border: 1px solid {c['ACCENT']};
            background-color: {c['HOVER']};
        }}

        QPushButton#dangerBtn {{
            background-color: transparent;
            color: {c['DANGER']};
            border: 1px solid {c['DANGER']};
            padding: 6px 18px;
        }}
        QPushButton#dangerBtn:hover {{
            background-color: {c['DANGER']};
            color: {c['BG']};
        }}

        /* ---- Checkboxes / radios ---- */
        QCheckBox, QRadioButton {{
            color: {c['TEXT']};
            spacing: 8px;
            padding: 4px 2px;
        }}
        QCheckBox::indicator, QRadioButton::indicator {{
            width: 18px; height: 18px;
            border: 1px solid {c['BORDER']};
            background: {c['PANEL_ALT']};
        }}
        QCheckBox::indicator {{ border-radius: 4px; }}
        QRadioButton::indicator {{ border-radius: 9px; }}
        QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
            background-color: {c['ACCENT']};
            border: 1px solid {c['ACCENT']};
        }}
        QCheckBox:hover, QRadioButton:hover {{
            color: {c['ACCENT']};
        }}

        /* ---- Footer ---- */
        QFrame#footer {{
            background-color: {c['BG']};
            border-top: 1px solid {c['BORDER']};
        }}
        """


# ============================================================
# Janela principal — com suporte ao módulo de expansão (16 canais)
# ============================================================
class EEGCollectorWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — {APP_EDITION} v{APP_VERSION}")
        self.resize(1600, 980)
        # Tamanho mínimo evita o layout quebrar quando o usuário encolhe a janela
        self.setMinimumSize(1100, 720)
        # Aceita arrastar e soltar arquivos CSV sobre a janela
        self.setAcceptDrops(True)

        # Configuração persistida (tema, mapping, nome de sessão, snapshots)
        self.config = AppConfig()
        # Aplica tema carregado em COLORS (in-place) antes de construir a UI
        _apply_theme_colors(self.config.theme)
        # Aplica idioma persistido (afeta tr() em todo o app)
        I18N.set_language(getattr(self.config, "language", "pt"))
        # Sessão atual
        self.current_session_dir = None
        self.session_name        = None
        self._snapshot_counter   = 0

        # Estado
        self.serial_thread  = None
        self.is_recording   = False
        self.csv_file       = None
        self.csv_writer     = None
        self.log_file       = None
        self.session_start  = None
        self.last_accel     = np.zeros(3)
        self.num_channels   = BASE_CHANNELS                # ativo agora (8 ou 16)
        self.daisy_enabled  = False                         # toggle do usuário
        self.channel_active = [True] * MAX_CHANNELS
        self.pending_marker = None

        # Buffers sempre dimensionados para MAX_CHANNELS
        self.buffer        = np.zeros((MAX_CHANNELS, BUFFER_SIZE), dtype=np.float64)
        self.buffer_pos    = 0
        self.samples_total = 0
        self.accel_buffer  = np.zeros((3, ACCEL_BUFFER_SIZE), dtype=np.float64)
        self.accel_pos     = 0
        self.spec_buffer   = np.full((MAX_CHANNELS, SPEC_FMAX, SPEC_FRAMES), -80.0)
        self.spec_pos      = 0

        self.markers = deque(maxlen=1000)
        self.filters = FilterChain(SAMPLE_RATE, MAX_CHANNELS)
        self.udp     = UDPSender()
        self.lsl     = LSLSender()

        # Cadastro de voluntários (portado do Data acquisition system)
        self.volunteers = VolunteerRegistry(self.config.save_directory)
        self.events_logger = EventsLogger()

        # Audit log estruturado (events.jsonl) — anexado ao iniciar gravação
        self._audit_fp = None

        # Sample timing audit — mede jitter inter-amostras (validação clínica)
        self._last_sample_t = None       # timestamp da amostra anterior
        self._dt_window     = deque(maxlen=250)   # últimos 1s de Δt
        self._dt_total      = 0          # total de amostras válidas medidas
        self._dt_mean       = 0.0
        self._dt_jitter     = 0.0
        self._dropped_count = 0

        pg.setConfigOption("background", COLORS["background"])
        pg.setConfigOption("foreground", COLORS["text"])
        pg.setConfigOptions(antialias=True)

        self._build_ui()
        self._refresh_ports()
        # Aplica mapeamento de canais carregado do config no Head Plot
        if hasattr(self, "head_plot"):
            self.head_plot.set_mapping(self.config.channel_mapping)
        # Aplica visibilidade inicial dos canais 9-16 (escondidos por padrão)
        self._apply_channel_visibility()
        # Aplica fontes nos eixos do pyqtgraph (JetBrains Mono nos tick labels
        # de valores numéricos; Inter nos rótulos descritivos dos eixos)
        self._apply_plot_fonts()
        # *** Garantia anti-flash: re-aplica o stylesheet global e os estilos
        # inline após toda a UI estar montada. Resolve casos em que o tema
        # salvo é diferente do default e o header pisca com a cor antiga.
        app = QtWidgets.QApplication.instance()
        if app:
            # Idempotente: re-aplicar a MESMA folha de estilo a uma janela ja
            # construida forca o "polish" de toda a arvore de widgets (64 plots
            # + 36 abas = ~17s!). So re-aplica se realmente mudou (ex.: tema
            # diferente do que o main() ja aplicou antes de construir a janela).
            _qss = build_stylesheet(COLORS)
            if app.styleSheet() != _qss:
                app.setStyleSheet(_qss)
        self._reapply_themed_inline_styles()
        # Recursos de usabilidade (status bar, menu Ajuda, atalhos, drag&drop)
        self._setup_usability()

        # Choice vindo do LauncherScreen (aplicado por apply_launcher_choice)
        self._pending_launcher_choice = None

        self.plot_timer = QTimer(self); self.plot_timer.timeout.connect(self._update_plots); self.plot_timer.start(50)
        self.analysis_timer = QTimer(self); self.analysis_timer.timeout.connect(self._update_analysis); self.analysis_timer.start(500)
        self.spec_timer = QTimer(self); self.spec_timer.timeout.connect(self._update_spectrogram); self.spec_timer.start(250)
        self.topo_timer = QTimer(self); self.topo_timer.timeout.connect(self._update_topology); self.topo_timer.start(500)
        # Timer de qualidade de sinal por canal (LEDs no header)
        self.quality_timer = QTimer(self); self.quality_timer.timeout.connect(self._update_channel_quality); self.quality_timer.start(500)
        self.layout_timer = QTimer(self); self.layout_timer.timeout.connect(self._update_layout_slots); self.layout_timer.start(200)

        sc = QtGui.QShortcut(QtGui.QKeySequence("M"), self)
        def _marker_hotkey():
            # Nao roubar a tecla 'M' quando o foco esta num campo de texto
            # (senao digitar "motor"/"estimulo"/hex injeta markers espurios).
            fw = QtWidgets.QApplication.focusWidget()
            if isinstance(fw, (QtWidgets.QLineEdit, QtWidgets.QTextEdit,
                               QtWidgets.QPlainTextEdit)) or \
               (isinstance(fw, QtWidgets.QComboBox) and fw.isEditable()):
                return
            self._inject_marker_text("M")
        sc.activated.connect(_marker_hotkey)

        # Se o idioma persistido não for pt, retraduz toda a UI já na primeira
        # pintura. (As strings hardcoded são em pt; tr() só converte se houver
        # entrada no dicionário I18N.)
        if I18N.current != "pt":
            try:
                self._retranslate_visible_ui()
            except Exception:
                pass

    # ==================================================================
    # UI build
    # ==================================================================
    def eventFilter(self, obj, event):
        # Redimensiona overlay de simulação quando o conteúdo central muda
        if (event.type() == QtCore.QEvent.Type.Resize and
                hasattr(self, "_sim_overlay") and obj is getattr(self, "tabs", None)):
            self._sim_overlay.resize(self.tabs.size())
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            self._update_header_responsive()
        except Exception:
            pass

    def _update_header_responsive(self):
        """Header responsivo: quando a barra não cabe, esconde a telemetria
        menos crítica (nesta ordem: acelerômetro → amostras → separador → LEDs
        por canal), preservando a MARCA/título e o resumo agregado 'Sinal OK'.
        Evita corte/sobreposição do título em telas/headers estreitos."""
        hw = getattr(self, "header_widget", None)
        if hw is None:
            return
        optional = [w for w in (
            getattr(self, "accel_label", None),
            getattr(self, "samples_label", None),
            getattr(self, "_header_sep", None),
            getattr(self, "quality_widget", None),
        ) if w is not None]
        for w in optional:
            w.setVisible(True)
        avail = hw.width()
        if avail <= 0:
            return
        for w in optional:
            if hw.sizeHint().width() <= avail:
                break
            w.setVisible(False)

    def _update_simulation_overlay(self, mode_text=""):
        """Sinaliza o modo Simulação/Playback SEM watermark sobre a tela.

        A pedido do usuário, o texto diagonal ('SIMULAÇÃO - DADOS NÃO REAIS'
        / 'PLAYBACK - REPRODUÇÃO DE SESSÃO') foi removido — ele atrapalhava a
        leitura dos parâmetros e do sinal. O modo continua claramente
        indicado pelo banner colorido no cabeçalho (_update_mode_banner),
        que não cobre nenhum conteúdo.
        """
        # Mantém o overlay sempre desarmado/escondido.
        self._sim_overlay_armed = False
        self._refresh_sim_overlay()

    def _refresh_sim_overlay(self, *_):
        """Watermark desativado — mantém o overlay sempre invisível."""
        if not hasattr(self, "_sim_overlay"):
            return
        self._sim_overlay.setVisible(False)

    @staticmethod
    def _wrap_scroll(content):
        """Envolve um QWidget em QScrollArea (vertical conforme necessário).

        Usado por abas densas (Conexão, EMG, ECG, EoG, Hardware, Acelerômetro)
        para que o conteúdo seja navegável em telas pequenas sem comer widgets.
        """
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        return scroll

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10); outer.setSpacing(10)
        outer.addWidget(self._build_header())

        # Estrutura hierárquica: 4 grupos top-level com sub-abas internas.
        # Reduz de 17 abas em fila → 4 grupos × 2-5 sub-abas (cada uma cabe
        # confortavelmente sem rolagem horizontal).
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("mainTabs")
        self.tabs.setMovable(False)

        def _make_subtabs():
            sub = QtWidgets.QTabWidget()
            sub.setObjectName("subTabs")
            sub.setDocumentMode(True)
            sub.setMovable(True)
            return sub

        # ===== GRUPO 1: Configurar (preparação antes da coleta) =====
        # Abas densas (Conexão, Filtros, Hardware, Calibração) recebem
        # QScrollArea para permitir scroll quando a tela for menor.
        g_setup = _make_subtabs()
        g_setup.addTab(self._build_volunteers_tab(),                tr("Voluntários"))
        g_setup.addTab(self._wrap_scroll(self._build_connection_tab()),  tr("Conexão"))
        g_setup.addTab(self._wrap_scroll(self._build_filters_tab()),     tr("Filtros e Canais"))
        g_setup.addTab(self._wrap_scroll(self._build_hardware_tab()),    tr("Hardware"))
        g_setup.addTab(self._wrap_scroll(self._build_calibration_tab()), tr("Calibração"))
        self.tabs.addTab(g_setup, tr("Configurar"))

        # ===== GRUPO 2: Visualizar (durante a coleta) =====
        # Organizado por modalidade:
        #   EEG  -> Tempo Real, Topografia, Espectrograma, Histórico, Layout Custom
        #   BIO  -> Multimodal (EMG, ECG, EoG) — agrupados em uma sub-aba com QTabWidget
        g_view = _make_subtabs()
        g_view.addTab(self._build_realtime_tab(),    tr("Tempo Real"))
        g_view.addTab(self._build_topology_tab(),    tr("Topografia"))
        g_view.addTab(self._build_spectrogram_tab(), tr("Espectrograma"))
        g_view.addTab(self._build_bio_multimodal_tab(), tr("Bio (EMG/ECG/EoG)"))
        g_view.addTab(self._build_history_tab(),     tr("Histórico"))
        g_view.addTab(self._build_layout_tab(),      tr("Layout Custom"))
        self.tabs.addTab(g_view, tr("Visualizar"))

        # ===== GRUPO 3: Analisar (após a coleta) =====
        # Adicionadas saídas interativas (BCI applications):
        #   Focus/SSVEP   -> análise de banda alpha/beta -> índice de foco
        #   EMG Joystick  -> 4 canais EMG mapeados para X/Y de joystick virtual
        g_anal = _make_subtabs()
        g_anal.addTab(self._build_analysis_tab(),     tr("Análises"))
        g_anal.addTab(self._build_offline_tab(),      tr("Offline"))
        g_anal.addTab(self._build_erp_tab(),          tr("ERP"))
        g_anal.addTab(self._build_connectivity_tab(), tr("Conectividade"))
        g_anal.addTab(self._build_ersd_tab(),         tr("ERS/ERD"))
        g_anal.addTab(self._wrap_scroll(self._build_focus_tab()),        tr("Focus / SSVEP"))
        g_anal.addTab(self._wrap_scroll(self._build_emg_joystick_tab()), tr("EMG Joystick"))
        g_anal.addTab(self._wrap_scroll(self._build_bci_trainer_tab()),  tr("BCI Trainer (MI)"))
        self.tabs.addTab(g_anal, tr("Analisar"))

        # ===== GRUPO 4: Sistema (integração + configurações) =====
        g_sys = _make_subtabs()
        g_sys.addTab(self._build_network_tab(),                       tr("Rede e Eventos"))
        g_sys.addTab(self._wrap_scroll(self._build_settings_tab()),   tr("Configurações"))
        self.tabs.addTab(g_sys, tr("Sistema"))

        # Default: começa em Configurar → Conexão (workflow natural)
        self.tabs.setCurrentIndex(0)
        g_setup.setCurrentIndex(1)

        outer.addWidget(self.tabs)
        # Guarda referências para uso futuro (ex.: navegar via código)
        self._main_tabs = self.tabs
        self._sub_tabs = {
            "setup": g_setup, "view": g_view,
            "analyse": g_anal, "system": g_sys,
        }

        # Overlay de simulação (oculto por padrão; aparece SÓ na aba Visualizar
        # quando o modo é Simulação/Playback — nunca sobre as telas de config).
        self._sim_overlay = _SimulationOverlay(self.tabs)
        self._sim_overlay.setVisible(False)
        self._sim_overlay.resize(self.tabs.size())
        self._sim_overlay_armed = False
        # Re-posiciona ao redimensionar a janela
        self.tabs.installEventFilter(self)
        # Reavalia a visibilidade do watermark ao trocar de aba principal
        self.tabs.currentChanged.connect(self._refresh_sim_overlay)

    # ---- Header ----
    def _build_header(self):
        self.header_widget = QtWidgets.QWidget()
        self.header_widget.setObjectName("header")
        self.header_widget.setAutoFillBackground(True)
        # Altura mínima ampliada para evitar que counters do header sejam
        # "comidos" pelo layout em telas pequenas.
        self.header_widget.setMinimumHeight(62)
        self.header_widget.setStyleSheet(
            f"#header {{ background-color: {COLORS['surface']}; "
            f"border: 1px solid {COLORS['border']}; border-radius: 4px; }}")
        layout = QtWidgets.QHBoxLayout(self.header_widget)
        layout.setContentsMargins(8, 6, 8, 6); layout.setSpacing(4)
        self.ufes_logo_lbl = None
        if os.path.exists(LOGO_UFES_PATH):
            self.ufes_logo_lbl = QtWidgets.QLabel()
            self.ufes_logo_lbl.setStyleSheet("background: transparent;")
            self._refresh_ufes_logo_pixmap()
            layout.addWidget(self.ufes_logo_lbl)

        self.title_label = QtWidgets.QLabel(f"◢ {APP_NAME.upper()}")
        # Fonte via setFont (e SEM letter-spacing) para o sizeHint refletir a
        # largura real — assim a marca não corta nem transborda sobre os badges.
        self.title_label.setFont(
            QtGui.QFont(FONT_UI, 15, QtGui.QFont.Weight.Bold))
        self.title_label.setStyleSheet(
            f"color: {COLORS['accent']}; background: transparent; padding: 2px 0;")
        self.title_label.setMinimumHeight(34)
        # Sem min-width: quando o header enche, o título corta graciosamente
        # (clip do próprio QLabel) em vez de sobrepor os badges ao lado.
        layout.addWidget(self.title_label)
        layout.addStretch()

        # Indicador de voluntário ativo
        self.volunteer_label = QtWidgets.QLabel("nenhum")
        self.volunteer_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-weight: bold; padding: 4px 10px;"
            f"border: 1px solid {COLORS['border']}; border-radius: 3px;")
        self.volunteer_label.setMinimumHeight(28)
        self.volunteer_label.setToolTip("Voluntário ativo — sessões vão para "
                                         "volunteers/<VID_Nome>/")
        layout.addWidget(self.volunteer_label)

        # Banner de aviso de modo (Simulação / Playback)
        self.mode_banner_label = QtWidgets.QLabel("")
        self.mode_banner_label.setVisible(False)
        self.mode_banner_label.setToolTip(
            "Modo atual de aquisição. Sinais em modo SIMULAÇÃO "
            "não são reais — apenas para teste da interface.")
        layout.addWidget(self.mode_banner_label)

        # Indicador de expansão
        self.expansion_label = QtWidgets.QLabel("8ch")
        self.expansion_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-weight: bold; padding: 4px 8px;"
            f"border: 1px solid {COLORS['border']}; border-radius: 3px;")
        self.expansion_label.setMinimumHeight(28)
        layout.addWidget(self.expansion_label)

        self.status_dot = QtWidgets.QLabel("●")
        self.status_dot.setStyleSheet(f"color: {COLORS['error']}; font-size: 18pt; padding: 0 4px;")
        layout.addWidget(self.status_dot)

        self.status_label = QtWidgets.QLabel("DESCONECTADO")
        self.status_label.setStyleSheet(
            f"color: {COLORS['error']}; font-weight: bold; "
            f"padding: 4px 8px; font-size: 11pt;")
        self.status_label.setMinimumHeight(28)
        layout.addWidget(self.status_label)

        self._header_sep = QtWidgets.QLabel("|")
        self._header_sep.setStyleSheet(f"color: {COLORS['border']};")
        layout.addWidget(self._header_sep)

        self.samples_label = QtWidgets.QLabel("Amostras: 0")
        self.samples_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; padding: 2px 5px; font-family: {FONT_DATA_STACK};")
        self.samples_label.setMinimumHeight(28)
        layout.addWidget(self.samples_label)

        self.accel_label = QtWidgets.QLabel("g: -.--  -.--  -.--")
        self.accel_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; padding: 2px 5px; font-family: {FONT_DATA_STACK};")
        self.accel_label.setMinimumHeight(28)
        layout.addWidget(self.accel_label)

        # LEDs de qualidade de sinal por canal (verde/amarelo/vermelho)
        self.quality_widget = QtWidgets.QWidget()
        ql_layout = QtWidgets.QHBoxLayout(self.quality_widget)
        ql_layout.setContentsMargins(4, 0, 4, 0); ql_layout.setSpacing(2)
        self.quality_leds = []
        for i in range(MAX_CHANNELS):
            led = QtWidgets.QLabel("●")
            led.setStyleSheet("color: #444; font-size: 10pt;")
            led.setToolTip(f"CH{i+1}: aguardando dados")
            led.setVisible(i < BASE_CHANNELS)
            self.quality_leds.append(led)
            ql_layout.addWidget(led)
        self.quality_widget.setToolTip(
            "Qualidade do sinal por canal — forma+cor: ● OK, ▲ ruidoso, ■ ruim/saturado"
        )
        layout.addWidget(self.quality_widget)

        # Semáforo agregado de prontidão do sinal (cor + símbolo + rótulo — WCAG)
        self.quality_summary = QtWidgets.QLabel("Sinal: —")
        self.quality_summary.setStyleSheet(
            "color: #888; font-size: 9pt; font-weight: bold; padding: 0 8px;")
        self.quality_summary.setToolTip(
            "Prontidão geral do sinal (resumo de todos os canais ativos).")
        layout.addWidget(self.quality_summary)

        # Indicador de qualidade temporal (Δt e jitter) — validação clínica
        self.timing_label = QtWidgets.QLabel("Δt: -.-- ms  ±-.-- ms")
        self.timing_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; padding: 2px 5px; font-family: {FONT_DATA_STACK};")
        self.timing_label.setMinimumHeight(28)
        self.timing_label.setToolTip(
            "Δt = média do intervalo entre amostras (esperado: "
            f"{1000.0/SAMPLE_RATE:.1f} ms a {SAMPLE_RATE} Hz)\n"
            "Jitter = desvio padrão de Δt (menor = melhor)\n"
            "Dropped = pacotes perdidos detectados"
        )
        layout.addWidget(self.timing_label)

        self.rec_indicator = QtWidgets.QLabel("")
        self.rec_indicator.setStyleSheet(f"color: {COLORS['error']}; font-weight: bold;")
        layout.addWidget(self.rec_indicator)

        bion_pm = self._load_bionica_pixmap()
        if bion_pm is not None:
            lbl = QtWidgets.QLabel(); lbl.setPixmap(bion_pm)
            lbl.setStyleSheet("background: transparent;"); layout.addWidget(lbl)
        return self.header_widget

    def _load_bionica_pixmap(self):
        if not os.path.exists(LOGO_BIONICA_PATH): return None
        img = QtGui.QImage(LOGO_BIONICA_PATH)
        if img.isNull(): return None
        img = img.convertToFormat(QtGui.QImage.Format.Format_ARGB32)
        # Torna pixels quase-brancos (fundo) transparentes. Vetorizado em
        # numpy — antes era um laço pixel-a-pixel que levava ~1,8 s.
        try:
            w, h = img.width(), img.height()
            ptr = img.bits(); ptr.setsize(img.sizeInBytes())
            stride = img.bytesPerLine()
            arr = np.frombuffer(ptr.asstring(), dtype=np.uint8) \
                    .reshape((h, stride // 4, 4)).copy()
            # ARGB32 little-endian -> ordem de bytes B, G, R, A
            b, g, r = arr[..., 0], arr[..., 1], arr[..., 2]
            mask = (r >= 235) & (g >= 235) & (b >= 235)
            arr[..., 3][mask] = 0  # alfa -> transparente
            img = QtGui.QImage(arr.tobytes(), w, h, stride,
                               QtGui.QImage.Format.Format_ARGB32).copy()
        except Exception:
            pass  # fallback: usa a imagem original sem processamento
        return QtGui.QPixmap.fromImage(img).scaledToHeight(
            40, QtCore.Qt.TransformationMode.SmoothTransformation)

    # ==================================================================
    # ABA VOLUNTÁRIOS — cadastro + ficha + histórico (subject mapping)
    # ==================================================================
    def _build_volunteers_tab(self):
        widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(10, 8, 10, 8); outer.setSpacing(6)

        header = QtWidgets.QLabel(
            f"<span style='color:{COLORS['accent']}; font-size:14pt; font-weight:bold;'>"
            "Cadastro de Voluntários</span>  "
            f"<span style='color:{COLORS['text_dim']};'>"
            "Cada voluntário tem ficha demográfica + histórico. Sessões gravadas "
            "com um voluntário ativo vão para <code>volunteers/&lt;VID_Nome&gt;/</code> "
            "— habilita análise futura com mapeamento do sujeito."
            "</span>"
        )
        header.setTextFormat(QtCore.Qt.TextFormat.RichText)
        header.setWordWrap(True)
        outer.addWidget(header)

        # Botões de ação
        btns = QtWidgets.QHBoxLayout()
        new_btn = QtWidgets.QPushButton("Novo voluntário")
        new_btn.clicked.connect(self._volunteer_new_dialog)
        btns.addWidget(new_btn)
        self.vol_select_btn = QtWidgets.QPushButton("Selecionar como ativo")
        self.vol_select_btn.clicked.connect(self._volunteer_select_active)
        btns.addWidget(self.vol_select_btn)
        self.vol_clear_btn = QtWidgets.QPushButton("Limpar ativo")
        self.vol_clear_btn.clicked.connect(self._volunteer_clear_active)
        btns.addWidget(self.vol_clear_btn)
        self.vol_delete_btn = QtWidgets.QPushButton("Deletar voluntário")
        self.vol_delete_btn.setStyleSheet(
            f"color: {COLORS['error']}; border-color: {COLORS['error']};")
        self.vol_delete_btn.clicked.connect(self._volunteer_delete)
        btns.addWidget(self.vol_delete_btn)
        self.vol_import_btn = QtWidgets.QPushButton("Importar exame externo...")
        self.vol_import_btn.setToolTip(
            "Adiciona um arquivo de exame realizado externamente "
            "(CSV/EDF/TXT/FIF) à ficha do voluntário selecionado."
        )
        self.vol_import_btn.clicked.connect(self._volunteer_import_external)
        btns.addWidget(self.vol_import_btn)
        refresh_btn = QtWidgets.QPushButton("Atualizar lista")
        refresh_btn.clicked.connect(self._volunteer_refresh_table)
        btns.addWidget(refresh_btn)
        btns.addStretch()
        self.vol_active_lbl = QtWidgets.QLabel("Ativo: (nenhum)")
        self.vol_active_lbl.setStyleSheet(
            f"color: {COLORS['accent']}; font-weight: bold;")
        btns.addWidget(self.vol_active_lbl)
        outer.addLayout(btns)

        # Tabela de voluntários
        self.vol_table = QtWidgets.QTableWidget(0, 7)
        self.vol_table.setHorizontalHeaderLabels(
            ["VID", "Nome", "Idade", "Sexo", "Mão", "Sessões", "Cadastro"])
        self.vol_table.verticalHeader().setVisible(False)
        # Resize misto: Nome é largo (stretch) — demais ficam ao conteúdo
        hdr = self.vol_table.horizontalHeader()
        hdr.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)  # Nome
        hdr.setMinimumSectionSize(40)
        self.vol_table.setHorizontalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.vol_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.vol_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.vol_table.setWordWrap(False)
        self.vol_table.itemDoubleClicked.connect(
            lambda *_: self._volunteer_select_active())
        # Menu de clique-direito (mouse) — adicionar/selecionar/importar/deletar.
        self.vol_table.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.vol_table.customContextMenuRequested.connect(
            self._volunteer_context_menu)
        outer.addWidget(self.vol_table, stretch=2)

        # Histórico do voluntário selecionado
        hist_group = QtWidgets.QGroupBox("Histórico de sessões do voluntário selecionado")
        hl = QtWidgets.QVBoxLayout(hist_group)
        self.vol_history = QtWidgets.QTextEdit()
        self.vol_history.setReadOnly(True)
        self.vol_history.setMaximumHeight(160)
        hl.addWidget(self.vol_history)
        outer.addWidget(hist_group, stretch=1)

        self.vol_table.itemSelectionChanged.connect(self._volunteer_show_history)
        self._volunteer_refresh_table()
        return widget

    def _volunteer_refresh_table(self):
        self.volunteers.set_base_dir(self.config.save_directory)
        vols = self.volunteers.list_volunteers()
        # Mensagem amigável quando ainda não há voluntários
        if not vols:
            self.vol_table.setRowCount(1)
            for c in range(self.vol_table.columnCount()):
                if c == 0:
                    it = QtWidgets.QTableWidgetItem(
                        "Nenhum voluntário cadastrado ainda. "
                        "Clique em 'Novo voluntário' acima para começar.")
                    it.setForeground(QtGui.QColor(COLORS["text_dim"]))
                else:
                    it = QtWidgets.QTableWidgetItem("")
                it.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
                self.vol_table.setItem(0, c, it)
            self._volunteer_update_indicator()
            return
        self.vol_table.setRowCount(len(vols))
        for r, p in enumerate(vols):
            prog = self.volunteers.get_progress(p)
            n_exec = len(prog.get("executions", []))
            cells = [
                p.get("vid", "?"),
                p.get("nome", ""),
                str(p.get("idade", "")),
                str(p.get("sexo", "")),
                str(p.get("mao_dominante", "")),
                str(n_exec),
                (p.get("created_at", "") or "")[:10],
            ]
            for c, val in enumerate(cells):
                it = QtWidgets.QTableWidgetItem(val)
                it.setData(QtCore.Qt.ItemDataRole.UserRole, p.get("_dirname"))
                if c == 0:
                    it.setForeground(QtGui.QColor(COLORS["accent"]))
                self.vol_table.setItem(r, c, it)
        self._volunteer_update_indicator()

    def _volunteer_selected_dirname(self):
        rows = self.vol_table.selectionModel().selectedRows()
        if not rows:
            it = self.vol_table.currentItem()
            if it is None:
                return None
            return it.data(QtCore.Qt.ItemDataRole.UserRole)
        return self.vol_table.item(rows[0].row(), 0).data(
            QtCore.Qt.ItemDataRole.UserRole)

    def _volunteer_show_history(self):
        dn = self._volunteer_selected_dirname()
        if not dn:
            self.vol_history.clear(); return
        try:
            prof = None
            for p in self.volunteers.list_volunteers():
                if p.get("_dirname") == dn:
                    prof = p; break
            if not prof:
                self.vol_history.clear(); return
            prog = self.volunteers.get_progress(prof)
            lines = [
                f'<b style="color:{COLORS["accent"]}">{prof.get("vid")} — {prof.get("nome")}</b>',
                f'Idade: {prof.get("idade","?")} | Sexo: {prof.get("sexo","?")} | '
                f'Mão: {prof.get("mao_dominante","?")} | '
                f'Escolaridade: {prof.get("escolaridade","?")}',
                f'Profissão: {prof.get("profissao","?")} | '
                f'Sono: {prof.get("qualidade_sono","?")}',
            ]
            if prof.get("condicao"):
                lines.append(f'Condição: {prof.get("condicao")}')
            if prof.get("medicacao"):
                lines.append(f'Medicação: {prof.get("medicacao")}')
            lines.append("<hr>")
            execs = prog.get("executions", [])
            if execs:
                lines.append(f'<b>{len(execs)} sessão(ões):</b>')
                for e in execs[-15:]:
                    lines.append(
                        f'&nbsp;&nbsp;• {e.get("at","?")} — {e.get("session","?")} '
                        f'({e.get("samples",0)} amostras, {e.get("markers",0)} markers)')
            else:
                lines.append("<i>Nenhuma sessão gravada ainda.</i>")
            self.vol_history.setHtml("<br>".join(lines))
        except Exception as exc:
            self.vol_history.setPlainText(f"Erro: {exc}")

    def _volunteer_new_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Cadastrar novo voluntário")
        dlg.setMinimumWidth(460)
        form = QtWidgets.QFormLayout(dlg)

        # Cabeçalho explicativo no diálogo
        legend = QtWidgets.QLabel(
            "<span style='color:" + COLORS["text_dim"] + ";'>"
            "Campos com <b style='color:" + COLORS["error"] + ";'>*</b> "
            "são obrigatórios. Os demais são opcionais e podem ser "
            "preenchidos mais tarde.</span>"
        )
        legend.setTextFormat(QtCore.Qt.TextFormat.RichText)
        legend.setWordWrap(True)
        form.addRow(legend)

        vid_edit = QtWidgets.QLineEdit(self.volunteers.next_vid())
        # Campos obrigatórios marcados com asterisco vermelho
        ast_red = f"<span style='color:{COLORS['error']};'>*</span>"
        vid_lbl = QtWidgets.QLabel(f"VID {ast_red}:")
        vid_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        form.addRow(vid_lbl, vid_edit)
        inputs = {"vid": vid_edit}

        REQUIRED_FIELDS = {"nome"}  # campos obrigatórios além de vid

        for key, label, typ, opts in VOLUNTEER_PROFILE_FIELDS:
            if typ == "enum":
                w = QtWidgets.QComboBox()
                w.addItem("")
                w.addItems(opts)
            elif typ == "int":
                w = QtWidgets.QSpinBox(); w.setRange(0, 130); w.setSpecialValueText("")
            elif typ == "float":
                w = QtWidgets.QDoubleSpinBox(); w.setRange(0, 400)
                w.setDecimals(1); w.setSpecialValueText("")
            else:
                w = QtWidgets.QLineEdit()
            # Label com asterisco se obrigatório
            if key in REQUIRED_FIELDS:
                lbl = QtWidgets.QLabel(f"{label} {ast_red}:")
                lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
                form.addRow(lbl, w)
            else:
                form.addRow(label + ":", w)
            inputs[key] = w

        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Cadastrar")
        bb.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setText("Cancelar")
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)

        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        profile = {"vid": vid_edit.text().strip()}
        for key, _label, typ, _opts in VOLUNTEER_PROFILE_FIELDS:
            w = inputs[key]
            if isinstance(w, QtWidgets.QComboBox):
                profile[key] = w.currentText().strip()
            elif isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                profile[key] = w.value() if w.value() > 0 else ""
            else:
                profile[key] = w.text().strip()
        try:
            self.volunteers.create_volunteer(profile)
            self._log(f"Voluntário cadastrado: {profile['vid']} — {profile.get('nome')}")
            self._audit_event("volunteer_created", vid=profile["vid"],
                              nome=profile.get("nome"))
            self._volunteer_refresh_table()
            self._volunteer_update_indicator()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Erro", str(exc))

    def _volunteer_context_menu(self, pos):
        """Menu de clique-direito na tabela de voluntários (interação por mouse)."""
        row = self.vol_table.rowAt(pos.y())
        if row >= 0:
            self.vol_table.selectRow(row)
        has_sel = bool(self._volunteer_selected_dirname())
        menu = QtWidgets.QMenu(self)
        act_new = menu.addAction("Novo voluntário")
        act_sel = menu.addAction("Selecionar como ativo")
        act_imp = menu.addAction("Importar exame externo…")
        menu.addSeparator()
        act_del = menu.addAction("Deletar voluntário")
        for a in (act_sel, act_imp, act_del):
            a.setEnabled(has_sel)
        chosen = menu.exec(self.vol_table.viewport().mapToGlobal(pos))
        if chosen == act_new:
            self._volunteer_new_dialog()
        elif chosen == act_sel:
            self._volunteer_select_active()
        elif chosen == act_imp:
            self._volunteer_import_external()
        elif chosen == act_del:
            self._volunteer_delete()

    def _volunteer_select_active(self):
        dn = self._volunteer_selected_dirname()
        if not dn:
            QtWidgets.QMessageBox.information(self, "Voluntários",
                "Selecione um voluntário na tabela primeiro.")
            return
        try:
            prof = self.volunteers.select_volunteer(dn)
            self._log(f"Voluntário ativo: {prof.get('vid')} — {prof.get('nome')}")
            self._audit_event("volunteer_selected", vid=prof.get("vid"))
            self._volunteer_update_indicator()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Erro", str(exc))

    def _volunteer_clear_active(self):
        self.volunteers.clear_current()
        self._log("Voluntário ativo removido (sessões irão para pasta padrão)")
        self._volunteer_update_indicator()

    def _volunteer_import_external(self):
        """Importa um exame realizado externamente para a ficha do voluntário.

        Aceita: CSV / EDF / BDF / FIF / TXT.
        Copia para volunteers/<VID_Nome>/imported_<timestamp>/ com summary.json.
        """
        dn = self._volunteer_selected_dirname()
        if not dn:
            QtWidgets.QMessageBox.information(self, "Importar exame externo",
                "Selecione um voluntário na tabela primeiro.")
            return
        # Pega o profile
        prof = None
        for p in self.volunteers.list_volunteers():
            if p.get("_dirname") == dn:
                prof = p; break
        if not prof:
            QtWidgets.QMessageBox.warning(self, "Erro",
                "Não foi possível ler a ficha do voluntário selecionado.")
            return
        # Abre seletor de arquivo
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Selecionar exame externo",
            os.path.expanduser("~"),
            "Sinais biológicos (*.csv *.edf *.bdf *.fif *.txt *.dat);;Todos (*)"
        )
        if not path:
            return
        if not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, "Erro",
                f"Arquivo não encontrado: {path}")
            return
        # Diálogo: título + tipo + data + nota
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Detalhes do exame importado")
        dl = QtWidgets.QFormLayout(dlg)
        title_edit = QtWidgets.QLineEdit()
        title_edit.setText(os.path.splitext(os.path.basename(path))[0])
        title_edit.setPlaceholderText("Ex.: EEG - vigília de repouso 21/03/2026")
        dl.addRow("Título do exame:", title_edit)
        type_combo = QtWidgets.QComboBox()
        type_combo.addItems(["EEG", "EMG", "ECG", "EoG", "Polissonografia",
                             "Eletromiografia Clínica", "Outro"])
        dl.addRow("Tipo de exame:", type_combo)
        date_edit = QtWidgets.QDateEdit()
        date_edit.setCalendarPopup(True)
        date_edit.setDate(QtCore.QDate.currentDate())
        dl.addRow("Data do exame:", date_edit)
        source_edit = QtWidgets.QLineEdit()
        source_edit.setPlaceholderText("Ex.: Hospital XYZ / Clínica do Dr. Y")
        dl.addRow("Origem (opcional):", source_edit)
        note_edit = QtWidgets.QTextEdit()
        note_edit.setMaximumHeight(80)
        note_edit.setPlaceholderText("Observações clínicas / contexto do exame...")
        dl.addRow("Notas:", note_edit)
        # Botões
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        dl.addRow(btn_box)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        title = title_edit.text().strip() or os.path.splitext(os.path.basename(path))[0]
        exam_type = type_combo.currentText()
        exam_date = date_edit.date().toString("yyyy-MM-dd")
        source = source_edit.text().strip()
        note = note_edit.toPlainText().strip()

        # Cria diretório imported_<timestamp>/
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitiza título para path
        safe_title = re.sub(r'[^\w\-]+', '_', title)[:40]
        vol_base = os.path.join(self.volunteers.volunteers_dir, dn)
        target_dir = os.path.join(vol_base, f"imported_{ts}_{safe_title}")
        try:
            os.makedirs(target_dir, exist_ok=True)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Erro",
                f"Não foi possível criar pasta de destino:\n{exc}")
            return
        # Copia arquivo (e renomeia para data.<ext> se for CSV)
        ext = os.path.splitext(path)[1].lower()
        try:
            import shutil
            if ext in (".edf", ".bdf"):
                # Converte o exame clínico p/ CSV nativo -> fica ANALISÁVEL no
                # Offline (leitor tolerante a header quebrado). Guarda o original.
                dest = os.path.join(target_dir, "data.csv")
                edf_to_native_csv(path, dest)
                shutil.copy2(path, os.path.join(target_dir, f"original{ext}"))
            elif ext == ".csv":
                dest = os.path.join(target_dir, "data.csv")
                shutil.copy2(path, dest)
            else:
                dest = os.path.join(target_dir, f"data{ext}")
                shutil.copy2(path, dest)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Erro ao importar",
                f"Falha ao importar arquivo:\n{exc}")
            return
        # Cria summary.json
        try:
            summary = {
                "imported":      True,
                "title":         title,
                "exam_type":     exam_type,
                "exam_date":     exam_date,
                "source":        source,
                "notes":         note,
                "original_path": path,
                "imported_at":   datetime.now().isoformat(timespec="seconds"),
                "volunteer":     {
                    "vid":  prof.get("vid", ""),
                    "nome": prof.get("nome", ""),
                },
                "num_channels":  "?",
                "started_at":    f"{exam_date}T00:00:00",
            }
            with open(os.path.join(target_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self._log(f"Aviso: não foi possível escrever summary.json: {exc}", error=True)

        # Atualiza histórico e tabela
        self._volunteer_refresh_table()
        self._volunteer_show_history()
        # Audit
        self._audit_event("external_exam_imported",
                          volunteer=dn, title=title, exam_type=exam_type)
        QtWidgets.QMessageBox.information(
            self, "Exame importado",
            f"Exame '{title}' importado com sucesso para o voluntário "
            f"{prof.get('vid','?')} — {prof.get('nome','')}.\n\n"
            f"Pasta: {target_dir}\n\n"
            f"Você pode analisar este exame na aba <b>Analisar → Offline</b>."
        )
        self._log(f"Exame externo importado: {title} -> {target_dir}")

    def _volunteer_delete(self):
        dn = self._volunteer_selected_dirname()
        if not dn:
            QtWidgets.QMessageBox.information(self, "Deletar voluntário",
                "Selecione um voluntário na tabela primeiro.")
            return
        # Pega ficha para mostrar nome
        prof = None
        for p in self.volunteers.list_volunteers():
            if p.get("_dirname") == dn:
                prof = p; break
        nome = prof.get("nome", dn) if prof else dn
        n_sessions = len(self.volunteers.get_progress(prof).get("executions", []))
        msg = (f"Confirmar exclusão PERMANENTE de '{prof.get('vid','?')} — {nome}'?\n\n"
               f"Esta ação irá apagar:\n"
               f"  • Ficha do voluntário\n"
               f"  • Histórico de progresso\n"
               f"  • TODAS as {n_sessions} sessões gravadas (CSVs, eventos, snapshots, PDFs)\n\n"
               f"Esta operação NÃO pode ser desfeita.")
        confirm = QtWidgets.QMessageBox.question(
            self, "Deletar voluntário", msg,
            QtWidgets.QMessageBox.StandardButton.Yes |
            QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            self.volunteers.delete_volunteer(dn)
            self._log(f"Voluntário deletado: {nome}")
            self._audit_event("volunteer_deleted", dirname=dn, nome=nome)
            self._volunteer_refresh_table()
            self.vol_history.clear()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Erro", str(exc))

    def _volunteer_update_indicator(self):
        cur = self.volunteers.current()
        if cur:
            txt = f"{cur.get('vid')} {cur.get('nome','')[:18]}"
            self.volunteer_label.setText(txt)
            self.volunteer_label.setStyleSheet(
                f"color: {COLORS['accent']}; font-weight: bold; padding: 0 8px;"
                f"border: 1px solid {COLORS['accent']}; border-radius: 3px;")
            self.vol_active_lbl.setText(f"Ativo: {cur.get('vid')} — {cur.get('nome')}")
        else:
            self.volunteer_label.setText("nenhum")
            self.volunteer_label.setStyleSheet(
                f"color: {COLORS['text_dim']}; font-weight: bold; padding: 0 8px;"
                f"border: 1px solid {COLORS['border']}; border-radius: 3px;")
            self.vol_active_lbl.setText("Ativo: (nenhum)")

    def _open_guided_stats(self):
        """Abre o facilitador de estatística guiada (comparar grupos de sessões)."""
        GuidedStatsDialog(self).exec()

    def _open_recipe(self):
        """Abre a Área Maker (receitas de análise composáveis e salváveis)."""
        RecipeDialog(self).exec()

    def _open_intra_stats(self):
        """Compara as condições/classes DENTRO de uma sessão (genérico/multimodal)."""
        IntraSessionStatsDialog(self, session=getattr(self, "_ersd_data", None)).exec()

    # ==================================================================
    # ABA OFFLINE — visualizador de sessões gravadas + análise por janela
    # ==================================================================
    def _build_offline_tab(self):
        widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(8, 6, 8, 6); outer.setSpacing(4)

        header = QtWidgets.QLabel(
            f"<span style='color:{COLORS['accent']}; font-size:14pt; font-weight:bold;'>"
            "Modo Offline — Análise de Sessões Gravadas</span>  "
            f"<span style='color:{COLORS['text_dim']};'>"
            "Escolha uma sessão na lista. Arraste a região colorida no plot "
            "para selecionar uma janela; as análises (FFT/bandas/estatísticas) "
            "aparecem abaixo automaticamente."
            "</span>"
        )
        header.setTextFormat(QtCore.Qt.TextFormat.RichText)
        header.setWordWrap(True)
        outer.addWidget(header)

        # Splitter vertical: browser + viewer + analyses
        vsplit = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        vsplit.setHandleWidth(4)

        # ---- Topo: Browser de sessões ----
        browser = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(browser); bl.setContentsMargins(0, 0, 0, 0)
        ctrl = QtWidgets.QHBoxLayout()
        refresh_btn = QtWidgets.QPushButton("Atualizar lista")
        refresh_btn.clicked.connect(self._offline_refresh_sessions)
        ctrl.addWidget(refresh_btn)
        load_sel_btn = QtWidgets.QPushButton("Carregar selecionada")
        load_sel_btn.clicked.connect(self._offline_load_selected)
        ctrl.addWidget(load_sel_btn)
        load_man_btn = QtWidgets.QPushButton("Abrir data.csv manualmente...")
        load_man_btn.clicked.connect(self._offline_load_manual_csv)
        ctrl.addWidget(load_man_btn)
        edf_btn = QtWidgets.QPushButton("Abrir EDF/BDF (reparo automático)...")
        edf_btn.setToolTip(
            "Abre exames clínicos .edf/.bdf (ex.: iCelera, BioWave) e converte "
            "para análise. Leitor TOLERANTE: lê MESMO com o cabeçalho quebrado "
            "(ex.: acento no nome do paciente) que faz o EDFbrowser recusar.")
        edf_btn.clicked.connect(self._offline_open_edf)
        ctrl.addWidget(edf_btn)
        stats_btn = QtWidgets.QPushButton("Estatística guiada (comparar grupos)")
        stats_btn.setToolTip(
            "Compara grupos de sessões (ex.: Antes × Depois), escolhe o teste "
            "estatístico adequado, monta a tabela e explica o resultado — sem "
            "exigir conhecimento prévio de estatística.")
        stats_btn.clicked.connect(self._open_guided_stats)
        ctrl.addWidget(stats_btn)
        intra_btn = QtWidgets.QPushButton("Comparar condições da sessão")
        intra_btn.setToolTip(
            "Compara as condições/classes marcadas DENTRO de uma sessão (sejam "
            "quais forem — dorsi/plantar, mãos, etc.) pela métrica que você "
            "escolher (banda EEG, RMS, ERD%). Genérico/multimodal; escolhe o "
            "teste estatístico sozinho.")
        intra_btn.clicked.connect(self._open_intra_stats)
        ctrl.addWidget(intra_btn)
        recipe_btn = QtWidgets.QPushButton("Receitas de análise (Área Maker)")
        recipe_btn.setToolTip(
            "Área Maker: monte um pipeline de análise (métrica + banda + canais), "
            "rode em 1+ sessões e SALVE como receita .json reutilizável/compartilhável.")
        recipe_btn.clicked.connect(self._open_recipe)
        ctrl.addWidget(recipe_btn)
        # Botão ICA (MNE) — limpeza de artefatos
        self.offline_ica_btn = QtWidgets.QPushButton("Limpar artefatos (ICA)")
        self.offline_ica_btn.setToolTip(
            "Remove piscadas por ICA (FastICA em numpy puro — NÃO precisa de MNE "
            "nem 'pip install', funciona direto no .exe):\n"
            "  1. Bandpass 1-40 Hz\n  2. FastICA (até 15 componentes)\n"
            "  3. Detecta o componente ocular (frontal Fp1/Fp2 ou curtose)\n"
            "  4. Reconstrói e salva data_clean.csv na pasta da sessão")
        self.offline_ica_btn.clicked.connect(self._offline_run_ica)
        ctrl.addWidget(self.offline_ica_btn)
        ctrl.addStretch()
        self.offline_status_lbl = QtWidgets.QLabel("Nenhuma sessão carregada.")
        self.offline_status_lbl.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-style: italic;")
        ctrl.addWidget(self.offline_status_lbl)
        bl.addLayout(ctrl)

        self.offline_table = QtWidgets.QTableWidget(0, 7)
        self.offline_table.setHorizontalHeaderLabels(
            ["Data", "Voluntário", "Sessão", "Duração", "Canais", "Markers", "Tamanho"])
        self.offline_table.verticalHeader().setVisible(False)
        # Sessão (col 2) e Voluntário (col 1) podem ser longas → stretch.
        # Demais (Data/Duração/Canais/Markers/Tamanho) → ajusta ao conteúdo.
        hdr = self.offline_table.horizontalHeader()
        hdr.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)  # Voluntário
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)  # Sessão
        hdr.setMinimumSectionSize(50)
        self.offline_table.setHorizontalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.offline_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.offline_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.offline_table.setWordWrap(False)
        self.offline_table.itemDoubleClicked.connect(
            lambda *_: self._offline_load_selected())
        bl.addWidget(self.offline_table)
        vsplit.addWidget(browser)

        # ---- Meio: Viewer da sessão (montagem completa) ----
        viewer = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(viewer); vl.setContentsMargins(0, 0, 0, 0)
        self.offline_plot = pg.PlotWidget(enableMenu=False)
        self.offline_plot.showGrid(x=True, y=False, alpha=0.15)
        self.offline_plot.setMenuEnabled(False)
        self.offline_plot.setLabel("bottom", "Tempo", units="s")
        self.offline_plot.getAxis("left").setWidth(70)
        self.offline_plot.hideButtons()
        self.offline_curves = []
        for ch in range(MAX_CHANNELS):
            cur = self.offline_plot.plot(
                pen=pg.mkPen(CHANNEL_COLORS[ch], width=1.0))
            self.offline_curves.append(cur)
        # Região selecionável (janela de análise)
        self.offline_region = pg.LinearRegionItem(
            values=[0, 5], movable=True,
            brush=pg.mkBrush(COLORS["accent"] + "33"),  # 33 = 20% alpha
            pen=pg.mkPen(COLORS["accent"], width=1))
        self.offline_region.sigRegionChangeFinished.connect(
            self._offline_update_region_analysis)
        self.offline_plot.addItem(self.offline_region)
        # Linhas de markers e itens auxiliares (limpos a cada load)
        self._offline_marker_items = []
        vl.addWidget(self.offline_plot)

        # Info da janela selecionada
        info_row = QtWidgets.QHBoxLayout()
        self.offline_region_info = QtWidgets.QLabel("Janela selecionada: —")
        self.offline_region_info.setStyleSheet(
            f"color: {COLORS['accent']}; font-family: {FONT_DATA_STACK};")
        info_row.addWidget(self.offline_region_info)
        info_row.addStretch()
        self.offline_channel_combo = QtWidgets.QComboBox()
        for i in range(MAX_CHANNELS):
            self.offline_channel_combo.addItem(f"CH{i+1}")
        self.offline_channel_combo.currentIndexChanged.connect(
            self._offline_update_region_analysis)
        info_row.addWidget(QtWidgets.QLabel("FFT/Stats do canal:"))
        info_row.addWidget(self.offline_channel_combo)
        vl.addLayout(info_row)
        vsplit.addWidget(viewer)

        # ---- Base: Análises da janela ----
        # Antes: QHBoxLayout horizontal com 3 caixas lado a lado (estourava a tela
        # em monitores pequenos). Agora: QTabWidget com 3 abas → cada análise
        # ocupa toda a largura disponível.
        bottom = QtWidgets.QTabWidget()
        bottom.setDocumentMode(True)
        bottom.setObjectName("offlineAnalysisTabs")

        fft_box = QtWidgets.QWidget()
        fft_l = QtWidgets.QVBoxLayout(fft_box)
        fft_l.setContentsMargins(6, 6, 6, 6)
        self.offline_fft_plot = pg.PlotWidget(enableMenu=False)
        self.offline_fft_plot.showGrid(x=True, y=True, alpha=0.15)
        self.offline_fft_plot.setLabel("left", "Amplitude", units="µV")
        self.offline_fft_plot.setLabel("bottom", "Freq", units="Hz")
        self.offline_fft_plot.setXRange(0, 60)
        self.offline_fft_plot.setMenuEnabled(False)
        self.offline_fft_curve = self.offline_fft_plot.plot(
            pen=pg.mkPen(COLORS["accent"], width=1.4))
        for _, (lo, _h) in EEG_BANDS.items():
            self.offline_fft_plot.addItem(pg.InfiniteLine(
                pos=lo, angle=90,
                pen=pg.mkPen(COLORS["border"], style=QtCore.Qt.PenStyle.DashLine)))
        fft_l.addWidget(self.offline_fft_plot)
        bottom.addTab(fft_box, "FFT da seleção")

        bands_box = QtWidgets.QWidget()
        bands_l = QtWidgets.QVBoxLayout(bands_box)
        bands_l.setContentsMargins(6, 6, 6, 6)
        self.offline_bands_plot = pg.PlotWidget(enableMenu=False)
        self.offline_bands_plot.showGrid(x=False, y=True, alpha=0.15)
        self.offline_bands_plot.setLabel("left", "Potência", units="µV²/Hz")
        self.offline_bands_plot.setMenuEnabled(False)
        self.offline_bands_plot.getAxis("bottom").setTicks(
            [list(enumerate(EEG_BANDS.keys()))])
        self.offline_bands_bars = pg.BarGraphItem(
            x=list(range(len(EEG_BANDS))), height=[0.0] * len(EEG_BANDS),
            width=0.6, brush=COLORS["accent"], pen=pg.mkPen(COLORS["accent_dim"]))
        self.offline_bands_plot.addItem(self.offline_bands_bars)
        self.offline_bands_plot.setXRange(-0.5, len(EEG_BANDS) - 0.5)
        bands_l.addWidget(self.offline_bands_plot)
        bottom.addTab(bands_box, "Bandas EEG")

        stats_box = QtWidgets.QWidget()
        stats_l = QtWidgets.QVBoxLayout(stats_box)
        stats_l.setContentsMargins(6, 6, 6, 6)
        self.offline_stats_table = QtWidgets.QTableWidget(0, 4)
        self.offline_stats_table.setHorizontalHeaderLabels(
            ["Canal", "Média (µV)", "SD (µV)", "RMS (µV)"])
        self.offline_stats_table.verticalHeader().setVisible(False)
        self.offline_stats_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.offline_stats_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.offline_stats_table.verticalHeader().setDefaultSectionSize(22)
        stats_l.addWidget(self.offline_stats_table)
        bottom.addTab(stats_box, "Estatísticas por canal")

        vsplit.addWidget(bottom)
        vsplit.setSizes([180, 400, 280])
        outer.addWidget(vsplit, stretch=1)

        # Estado interno
        self._offline_data = None      # dict do _load_session_csv
        self._offline_scale_uV = 100.0 # escala visual do montage
        self._offline_session_info = None
        # Atualiza a lista na criação
        self._offline_refresh_sessions()
        return widget

    # ----- Listagem de sessões -----
    def _scan_session_dir(self, sess_dir):
        """Para uma pasta de sessão, retorna metadados se for válida."""
        csv_path = os.path.join(sess_dir, "data.csv")
        if not os.path.isfile(csv_path):
            return None
        info = {"path": sess_dir, "csv": csv_path,
                "name": os.path.basename(sess_dir),
                "size_kb": os.path.getsize(csv_path) // 1024,
                "volunteer": "", "channels": "?", "markers": 0,
                "duration_s": 0.0, "date": ""}
        # Lê summary.json se houver
        summary = os.path.join(sess_dir, "summary.json")
        if os.path.isfile(summary):
            try:
                with open(summary, "r", encoding="utf-8") as f:
                    s = json.load(f)
                info["channels"] = s.get("num_channels", "?")
                v = s.get("volunteer") or {}
                if isinstance(v, dict):
                    info["volunteer"] = f"{v.get('vid','')} {v.get('nome','')}".strip()
                info["date"] = (s.get("started_at", "") or "")[:19].replace("T", " ")
            except Exception: pass
        # Conta linhas do data.csv para duração + samples
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                header = f.readline()
                ch_cnt = sum(1 for h in header.split(",") if h.endswith("_uV"))
                if info["channels"] == "?":
                    info["channels"] = ch_cnt
                n_data = sum(1 for _ in f)
            sr = 250  # default
            info["duration_s"] = n_data / sr
        except Exception: pass
        # events.csv → conta markers
        ev_path = os.path.join(sess_dir, "events.csv")
        if os.path.isfile(ev_path):
            try:
                with open(ev_path, "r", encoding="utf-8") as f:
                    info["markers"] = max(0, sum(1 for _ in f) - 1)
            except Exception: pass
        return info

    def _list_all_sessions(self):
        """Vasculha save_directory (sessões soltas) + volunteers/ (por sujeito)."""
        results = []
        root = self.config.save_directory
        if not os.path.isdir(root):
            return results
        # Sessões soltas em <save_directory>/*/data.csv
        try:
            for entry in os.listdir(root):
                full = os.path.join(root, entry)
                if os.path.isdir(full) and entry != "volunteers":
                    info = self._scan_session_dir(full)
                    if info: results.append(info)
        except Exception: pass
        # Sessões por voluntário em <save_directory>/volunteers/<VID>/*/data.csv
        vols_dir = os.path.join(root, "volunteers")
        if os.path.isdir(vols_dir):
            for vol_entry in os.listdir(vols_dir):
                vfull = os.path.join(vols_dir, vol_entry)
                if not os.path.isdir(vfull): continue
                vol_label = vol_entry
                # Tenta o nome do profile.json
                pj = os.path.join(vfull, "profile.json")
                if os.path.isfile(pj):
                    try:
                        with open(pj, "r", encoding="utf-8") as f:
                            p = json.load(f)
                        vol_label = f"{p.get('vid', vol_entry)} {p.get('nome','')}".strip()
                    except Exception: pass
                for sess_entry in os.listdir(vfull):
                    sfull = os.path.join(vfull, sess_entry)
                    info = self._scan_session_dir(sfull)
                    if info:
                        if not info["volunteer"]:
                            info["volunteer"] = vol_label
                        results.append(info)
        # Ordena por data desc (depois por nome)
        results.sort(key=lambda r: (r.get("date", ""), r["name"]), reverse=True)
        return results

    def _offline_refresh_sessions(self):
        sessions = self._list_all_sessions()
        self.offline_table.setRowCount(len(sessions))
        for r, info in enumerate(sessions):
            dur = info["duration_s"]
            dur_txt = f"{int(dur//60)}min {int(dur%60):02d}s" if dur >= 60 else f"{dur:.1f}s"
            cells = [
                info.get("date", "") or "—",
                info.get("volunteer", "") or "—",
                info["name"],
                dur_txt,
                str(info["channels"]),
                str(info["markers"]),
                f"{info['size_kb']} KB" if info['size_kb'] < 1024 else f"{info['size_kb']/1024:.1f} MB",
            ]
            for c, val in enumerate(cells):
                it = QtWidgets.QTableWidgetItem(val)
                it.setData(QtCore.Qt.ItemDataRole.UserRole, info["csv"])
                if c == 0:
                    it.setForeground(QtGui.QColor(COLORS["accent"]))
                self.offline_table.setItem(r, c, it)
        self.offline_status_lbl.setText(
            f"{len(sessions)} sessão(ões) encontradas em {self.config.save_directory}"
        )

    def _offline_load_selected(self):
        rows = self.offline_table.selectionModel().selectedRows()
        if not rows:
            QtWidgets.QMessageBox.information(self, "Offline",
                "Selecione uma sessão na tabela primeiro.")
            return
        csv_path = self.offline_table.item(
            rows[0].row(), 0).data(QtCore.Qt.ItemDataRole.UserRole)
        self._offline_load_csv(csv_path)

    def _offline_run_ica(self):
        """Remove piscadas por ICA (FastICA em numpy PURO) na sessão carregada.
        NÃO precisa de MNE nem 'pip install' — funciona direto no .exe."""
        if not getattr(self, "_offline_data", None):
            QtWidgets.QMessageBox.information(
                self, "Sem sessão",
                "Carregue uma sessão primeiro (Carregar selecionada ou Abrir CSV)."
            )
            return
        d = self._offline_data
        # Confirmação (ICA pode demorar)
        n_ch = len(d.get("ch_names", []))
        n_samples = d["eeg"].shape[1] if hasattr(d.get("eeg"), "shape") else 0
        dur = n_samples / d.get("sr", SAMPLE_RATE) if n_samples else 0
        confirm = QtWidgets.QMessageBox.question(
            self, "Rodar ICA",
            f"<b>Pré-requisitos atendidos.</b><br><br>"
            f"Canais: {n_ch}<br>Duração: {dur:.1f} s<br>"
            f"Sample rate: {d.get('sr', SAMPLE_RATE)} Hz<br><br>"
            "ICA pode levar 5-60s. Recomendado pelo menos 60s de sinal.<br>"
            "Resultado salvo como <code>data_clean.csv</code>.<br><br>Continuar?",
            QtWidgets.QMessageBox.StandardButton.Yes |
            QtWidgets.QMessageBox.StandardButton.No)
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes: return
        try:
            self.offline_status_lbl.setText("Rodando ICA (FastICA numpy)...")
            self.offline_status_lbl.setStyleSheet(f"color: {COLORS['warning']};")
            QtWidgets.QApplication.processEvents()
            sr = d.get("sr", SAMPLE_RATE)
            cleaned_uV, excluded, info = ica_clean_eog(
                np.asarray(d["eeg"], dtype=float), sr, d["ch_names"])
            d["eeg"] = cleaned_uV
            d["ica_applied"] = True
            d["ica_excluded"] = excluded
            # Salva CSV limpo (formato nativo)
            try:
                base_dir = (os.path.dirname(d.get("csv_path") or "")
                            or d.get("path") or self.config.save_directory)
                clean_path = os.path.join(base_dir, "data_clean.csv")
                header = "time_s," + ",".join(f"{nm}_uV" for nm in d["ch_names"])
                t_axis = (np.arange(cleaned_uV.shape[1]) / sr).reshape(-1, 1)
                np.savetxt(clean_path, np.column_stack([t_axis, cleaned_uV.T]),
                           delimiter=",", header=header, comments="",
                           fmt=["%.6f"] + ["%.6g"] * n_ch)
                self._log(f"ICA aplicada: {len(excluded)} componente(s) removido(s). "
                          f"Salvo em {clean_path}")
            except Exception as exc:
                self._log(f"ICA OK em memória, mas falha ao salvar: {exc}", error=True)
            self._offline_update_region_analysis()
            self.offline_status_lbl.setText(
                f"ICA aplicada: {len(excluded)} componente(s) de piscada removido(s)."
                if excluded else
                "ICA rodou: nenhum componente ocular forte (sinal já parece limpo).")
            self.offline_status_lbl.setStyleSheet(
                f"color: {SIGNAL_TYPE_COLORS['EEG']}; font-weight: bold;")
            try:
                self._audit_event("ica_applied", excluded=excluded,
                                  n_components=info.get("n_components"))
            except Exception:
                pass
        except Exception as exc:
            self.offline_status_lbl.setText(f"Erro ICA: {exc}")
            self.offline_status_lbl.setStyleSheet(f"color: {COLORS['error']};")
            self._log(f"ICA falhou: {exc}", error=True)

    def _offline_load_manual_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Abrir data.csv para análise offline",
            self.config.save_directory, "CSV (*.csv);;Todos (*)")
        if path:
            self._offline_load_csv(path)

    def _offline_open_edf(self):
        """Abre um EDF/BDF (leitor tolerante), converte p/ CSV nativo e carrega
        no Offline. Lê exames clínicos MESMO com header não-ASCII quebrado."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Abrir exame EDF/BDF para análise",
            os.path.expanduser("~"),
            "EEG clínico (*.edf *.bdf *.EDF *.BDF);;Todos (*)")
        if not path:
            return
        try:
            base = os.path.join(self.config.save_directory, "importados_edf")
            os.makedirs(base, exist_ok=True)
            stem = re.sub(r"[^\w\-]+", "_",
                          os.path.splitext(os.path.basename(path))[0])[:50] or "exame"
            out_dir = os.path.join(
                base, f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(out_dir, exist_ok=True)
            out_csv = os.path.join(out_dir, "data.csv")
            _, labels, fs, n = edf_to_native_csv(path, out_csv)
            self._log(f"EDF importado: {os.path.basename(path)} -> "
                      f"{len(labels)} canais @ {fs} Hz, {n} amostras "
                      f"({n / fs:.0f}s).")
        except Exception as exc:
            logging.getLogger("eeg").exception("Falha lendo EDF")
            QtWidgets.QMessageBox.critical(
                self, "Erro ao ler EDF",
                f"Não foi possível ler o arquivo EDF/BDF:\n{exc}")
            return
        self._offline_load_csv(out_csv)         # reaproveita o pipeline offline
        QtWidgets.QMessageBox.information(
            self, "EDF importado",
            f"Exame lido e convertido com sucesso:\n\n"
            f"• {len(labels)} canais a {fs} Hz\n"
            f"• {n / fs:.0f} s de sinal\n\n"
            f"Carregado no modo Offline. Arraste a região no gráfico para "
            f"selecionar uma janela e ver FFT/bandas/estatísticas.")

    def _offline_load_csv(self, csv_path):
        """Carrega o CSV e popula o viewer com todos os canais (montage style)."""
        d = self._load_session_csv(csv_path)
        if not d:
            return
        self._offline_data = d
        sess_dir = os.path.dirname(csv_path)
        self._offline_session_info = {"path": sess_dir,
                                       "csv": csv_path,
                                       "name": os.path.basename(sess_dir)}
        eeg, sr, names = d["eeg"], d["sr"], d["ch_names"]
        n_ch, n_samp = eeg.shape
        t = np.arange(n_samp) / sr
        # Auto-scale: 4x maior SD
        sd_max = max(float(np.std(eeg[i])) for i in range(n_ch)) if n_ch else 1.0
        self._offline_scale_uV = max(10.0, sd_max * 4.0)

        # Desenha cada canal com offset vertical (top = canal 0)
        for ch in range(MAX_CHANNELS):
            if ch < n_ch:
                baseline = (n_ch - 1 - ch)
                y = (eeg[ch] - float(np.mean(eeg[ch]))) / self._offline_scale_uV + baseline
                self.offline_curves[ch].setData(t, y)
            else:
                self.offline_curves[ch].setData([], [])
        # Ticks com nomes
        ticks = [(n_ch - 1 - i, names[i]) for i in range(n_ch)]
        self.offline_plot.getAxis("left").setTicks([ticks])
        self.offline_plot.setYRange(-0.6, max(0.6, n_ch - 0.4), padding=0)
        self.offline_plot.setXRange(0, t[-1] if len(t) > 0 else 1, padding=0)

        # Limpa markers anteriores
        for it in self._offline_marker_items:
            try: self.offline_plot.removeItem(it)
            except Exception: pass
        self._offline_marker_items.clear()
        # Adiciona linhas para cada marker
        for tm, label in d["markers"]:
            mline = pg.InfiniteLine(
                pos=tm, angle=90,
                pen=pg.mkPen(COLORS["warning"], width=1,
                              style=QtCore.Qt.PenStyle.DashLine),
                label=label,
                labelOpts={"color": COLORS["warning"], "position": 0.92,
                            "movable": False})
            self.offline_plot.addItem(mline)
            self._offline_marker_items.append(mline)

        # Reseta combo de canal
        self.offline_channel_combo.blockSignals(True)
        self.offline_channel_combo.clear()
        for nm in names:
            self.offline_channel_combo.addItem(nm)
        self.offline_channel_combo.setCurrentIndex(0)
        self.offline_channel_combo.blockSignals(False)

        # Reposiciona região para os primeiros 10s
        end_x = min(10.0, t[-1] if len(t) > 0 else 5.0)
        self.offline_region.setRegion([0.0, end_x])

        # Status
        dur = n_samp / sr
        sr_info = self._format_sr_check(d)
        self.offline_status_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.offline_status_lbl.setText(
            f"✓ <b>{os.path.basename(sess_dir)}</b> · "
            f"{n_ch} canais · {dur:.1f}s · {len(d['markers'])} markers<br>"
            + sr_info
        )
        # Stats por canal
        self.offline_stats_table.setRowCount(n_ch)
        for ch in range(n_ch):
            for col in range(4):
                if self.offline_stats_table.item(ch, col) is None:
                    self.offline_stats_table.setItem(
                        ch, col, QtWidgets.QTableWidgetItem(""))
        # Análise inicial
        self._offline_update_region_analysis()
        self._audit_event("offline_load", path=csv_path)

    def _offline_update_region_analysis(self):
        if not self._offline_data: return
        d = self._offline_data
        eeg, sr, names = d["eeg"], d["sr"], d["ch_names"]
        t0, t1 = self.offline_region.getRegion()
        t0 = max(0.0, float(t0)); t1 = max(t0 + 0.1, float(t1))
        i0 = int(round(t0 * sr)); i1 = int(round(t1 * sr))
        i0 = max(0, i0); i1 = min(eeg.shape[1], i1)
        dur = (i1 - i0) / sr
        ch = min(self.offline_channel_combo.currentIndex(), len(names) - 1)
        if ch < 0: ch = 0
        # Atualiza info
        self.offline_region_info.setText(
            f"Janela: {t0:.2f}s – {t1:.2f}s  (dur {dur:.2f}s, {i1-i0} amostras) "
            f"| Canal FFT/Bandas: {names[ch] if ch < len(names) else 'CH?'}"
        )
        seg_ch = eeg[ch, i0:i1] if (i1 - i0) >= 2 else np.array([])
        # FFT
        if seg_ch.size >= 2:
            freqs, spec = SignalProcessor.compute_fft(seg_ch, sr)
            if freqs.size: self.offline_fft_curve.setData(freqs, spec)
            # Bandas
            powers = SignalProcessor.compute_band_powers(seg_ch, sr)
            self.offline_bands_bars.setOpts(height=list(powers.values()))
        else:
            self.offline_fft_curve.setData([], [])
            self.offline_bands_bars.setOpts(height=[0.0] * len(EEG_BANDS))
        # Stats por canal — garante linhas/itens suficientes p/ a sessao carregada
        # (sessao pode ter mais canais que as linhas iniciais da tabela -> item None).
        n_ch = eeg.shape[0]
        tbl = self.offline_stats_table
        if tbl.rowCount() < n_ch:
            tbl.setRowCount(n_ch)
        for c in range(n_ch):
            for col in range(4):
                if tbl.item(c, col) is None:
                    tbl.setItem(c, col, QtWidgets.QTableWidgetItem())
            seg = eeg[c, i0:i1] if (i1 - i0) >= 2 else np.array([])
            stats = SignalProcessor.compute_statistics(seg) if seg.size else {
                "mean": 0, "std": 0, "rms": 0}
            tbl.item(c, 0).setText(names[c] if c < len(names) else f"CH{c+1}")
            tbl.item(c, 0).setForeground(
                QtGui.QColor(CHANNEL_COLORS[c] if c < len(CHANNEL_COLORS) else "#fff"))
            tbl.item(c, 1).setText(f"{stats['mean']:+.2f}")
            tbl.item(c, 2).setText(f"{stats['std']:.2f}")
            tbl.item(c, 3).setText(f"{stats['rms']:.2f}")
            for col in (1, 2, 3):
                tbl.item(c, col).setTextAlignment(
                    QtCore.Qt.AlignmentFlag.AlignCenter)

    # ==================================================================
    # ABA ERS/ERD — análise de imagética motora a partir de CSV BCI
    # (compatível com arquivos do "Data acquisition system.py")
    # ==================================================================
    # Faixas típicas para motor imagery
    ERSD_BANDS = {
        "Mu (8-13 Hz)":   (8.0, 13.0),
        "Beta (13-30 Hz)":(13.0, 30.0),
        "Mu+Beta (8-30)": (8.0, 30.0),
        "Alpha (8-12 Hz)":(8.0, 12.0),
        "Gamma (30-50)":  (30.0, 50.0),
    }
    # Fases que servem como referência baseline para ERD%
    BASELINE_PHASES = ("baseline", "pre_rest", "inter_baseline")

    def _build_ersd_tab(self):
        widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(8, 6, 8, 6); outer.setSpacing(4)

        header = QtWidgets.QLabel(
            f"<span style='color:{COLORS['accent']}; font-size:14pt; font-weight:bold;'>"
            "ERS/ERD — Event-Related (De)Synchronization</span>  "
            f"<span style='color:{COLORS['text_dim']};'>"
            "Compatível com sessões do <i>Data acquisition system</i> (formato "
            "BCI: 15 canais EEG @ 125 Hz + events.csv com fases mi/baseline/...). "
            "ERD% &lt; 0 = dessincronização típica de motor imagery."
            "</span>"
        )
        header.setTextFormat(QtCore.Qt.TextFormat.RichText)
        header.setWordWrap(True)
        outer.addWidget(header)

        # Controles
        ctrl_group = QtWidgets.QGroupBox("Carregamento e parâmetros")
        cgl = QtWidgets.QGridLayout(ctrl_group)
        cgl.setHorizontalSpacing(8); cgl.setVerticalSpacing(4)

        self.ersd_load_btn = QtWidgets.QPushButton("Carregar CSV BCI...")
        self.ersd_load_btn.clicked.connect(self._ersd_load_csv)
        cgl.addWidget(self.ersd_load_btn, 0, 0)

        self.ersd_status = QtWidgets.QLabel(
            "Nenhum CSV carregado. Selecione um arquivo do Data acquisition system "
            "(deve ter coluna 'Event Id' + arquivo *_events.csv ao lado).")
        self.ersd_status.setStyleSheet(f"color: {COLORS['text_dim']};")
        self.ersd_status.setWordWrap(True)
        cgl.addWidget(self.ersd_status, 0, 1, 1, 5)

        cgl.addWidget(QtWidgets.QLabel("Classe:"), 1, 0)
        self.ersd_class_combo = QtWidgets.QComboBox()
        self.ersd_class_combo.addItem("Todas (média)", -1)
        self.ersd_class_combo.addItem("LEFT_HAND (mão E)", 0)
        self.ersd_class_combo.addItem("RIGHT_HAND (mão D)", 1)
        self.ersd_class_combo.addItem("DORSI (pé dorsi)", 2)
        self.ersd_class_combo.addItem("PLANTAR (pé plantar)", 3)
        cgl.addWidget(self.ersd_class_combo, 1, 1)

        cgl.addWidget(QtWidgets.QLabel("Banda:"), 1, 2)
        self.ersd_band_combo = QtWidgets.QComboBox()
        self.ersd_band_combo.addItems(list(self.ERSD_BANDS.keys()))
        self.ersd_band_combo.setCurrentText("Mu (8-13 Hz)")
        cgl.addWidget(self.ersd_band_combo, 1, 3)

        cgl.addWidget(QtWidgets.QLabel("Baseline:"), 1, 4)
        self.ersd_baseline_combo = QtWidgets.QComboBox()
        self.ersd_baseline_combo.addItem("baseline (preferido)", "baseline")
        self.ersd_baseline_combo.addItem("pre_rest", "pre_rest")
        self.ersd_baseline_combo.addItem("ambos (b/pre_rest)", "auto")
        self.ersd_baseline_combo.setCurrentIndex(2)
        cgl.addWidget(self.ersd_baseline_combo, 1, 5)

        self.ersd_compute_btn = QtWidgets.QPushButton("▶ Computar ERS/ERD")
        self.ersd_compute_btn.clicked.connect(self._ersd_compute)
        self.ersd_compute_btn.setEnabled(False)
        cgl.addWidget(self.ersd_compute_btn, 2, 0, 1, 2)

        self.ersd_info_label = QtWidgets.QLabel("—")
        self.ersd_info_label.setStyleSheet(f"color: {COLORS['accent']};")
        cgl.addWidget(self.ersd_info_label, 2, 2, 1, 4)
        outer.addWidget(ctrl_group)

        # Splitter principal: topografia + barras + curso temporal
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setHandleWidth(4)

        # Esquerda: head plot topográfico
        topo_box = QtWidgets.QGroupBox("Topografia ERD% (mapa de calor)")
        tl = QtWidgets.QVBoxLayout(topo_box)
        self.ersd_head = HeadPlotWidget()
        # Mapeamento dos canais do BCI (15 canais) no head plot
        bci_map = self.BCI_EEG_CHANNELS + [""] * (MAX_CHANNELS - len(self.BCI_EEG_CHANNELS))
        self.ersd_head.set_mapping(bci_map)
        self.ersd_head.set_num_channels(15)
        tl.addWidget(self.ersd_head)
        split.addWidget(topo_box)

        # Direita: barras ERD% por canal + curso temporal
        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        bar_box = QtWidgets.QGroupBox("ERD% por canal (média entre trials)")
        bl = QtWidgets.QVBoxLayout(bar_box)
        self.ersd_bar_plot = pg.PlotWidget(enableMenu=False)
        self.ersd_bar_plot.showGrid(x=False, y=True, alpha=0.2)
        self.ersd_bar_plot.setLabel("left", "ERD%", units="")
        self.ersd_bar_plot.setMenuEnabled(False)
        self.ersd_bars = pg.BarGraphItem(
            x=list(range(15)), height=[0.0] * 15, width=0.6,
            brushes=[pg.mkBrush(CHANNEL_COLORS[i % MAX_CHANNELS]) for i in range(15)])
        self.ersd_bar_plot.addItem(self.ersd_bars)
        self.ersd_bar_plot.getAxis("bottom").setTicks(
            [list(enumerate(self.BCI_EEG_CHANNELS))])
        # Linha em y=0
        self.ersd_bar_plot.addItem(pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen(COLORS["text_dim"], width=1)))
        bl.addWidget(self.ersd_bar_plot)
        right.addWidget(bar_box)

        time_box = QtWidgets.QGroupBox("Curso temporal ERD%(t) — canal C3 vs C4")
        tml = QtWidgets.QVBoxLayout(time_box)
        self.ersd_time_plot = pg.PlotWidget(enableMenu=False)
        self.ersd_time_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ersd_time_plot.setLabel("left", "ERD%", units="")
        self.ersd_time_plot.setLabel("bottom", "Tempo relativo ao MI", units="s")
        self.ersd_time_plot.setMenuEnabled(False)
        self.ersd_time_plot.addLegend(offset=(10, 10))
        self.ersd_c3_curve = self.ersd_time_plot.plot(
            pen=pg.mkPen(COLORS["accent"], width=2), name="C3")
        self.ersd_c4_curve = self.ersd_time_plot.plot(
            pen=pg.mkPen("#ff5599", width=2), name="C4")
        self.ersd_cz_curve = self.ersd_time_plot.plot(
            pen=pg.mkPen("#5599ff", width=2), name="Cz")
        # Linha vertical em t=0 (início do MI)
        self.ersd_time_plot.addItem(pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen(COLORS["warning"], width=1,
                          style=QtCore.Qt.PenStyle.DashLine)))
        self.ersd_time_plot.addItem(pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(COLORS["text_dim"], width=1)))
        tml.addWidget(self.ersd_time_plot)
        right.addWidget(time_box)
        right.setSizes([300, 300])
        split.addWidget(right)
        split.setSizes([520, 580])
        outer.addWidget(split, stretch=1)

        # Estado interno
        self._ersd_data = None  # dict carregado
        return widget

    def _ersd_load_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Abrir CSV do Data acquisition system",
            self.config.save_directory, "CSV (*.csv);;Todos (*)")
        if not path:
            return
        d = self._load_session_csv(path)
        if not d:
            return
        if d.get("format") != "bci_protocol":
            QtWidgets.QMessageBox.warning(self, "Formato",
                "Este CSV não parece ser do Data acquisition system "
                "(faltam colunas 'Event Id' / 'Class Id'). "
                "Para análise ERD genérica, use a aba Offline.")
            return
        trials = d.get("trials") or []
        if not trials:
            QtWidgets.QMessageBox.warning(self, "Sem trials detectados",
                "CSV BCI carregado, mas não foi possível extrair fases nem "
                "do *_events.csv companheiro nem da coluna 'Event Id' (toda "
                "zerada). Sem fases não há como calcular ERD.")
            return
        # Aviso (não bloqueia) se não tem events.csv: trials reconstruídos
        if not d.get("events_csv_path"):
            self._log("Aviso: sem _events.csv companheiro — trials "
                      "reconstruídos a partir da coluna 'Event Id'.")
        self._ersd_data = d
        # Sumário
        n_mi = sum(1 for t in trials if t.get("phase") == "mi")
        n_base = sum(1 for t in trials if t.get("phase") in self.BASELINE_PHASES)
        classes_present = sorted({t.get("class_id") for t in trials
                                   if t.get("phase") == "mi" and t.get("class_id", -1) >= 0})
        cls_str = ", ".join(self.BCI_CLASS_NAMES.get(c, "?") for c in classes_present)
        n_ch = d["eeg"].shape[0]; sr = d["sr"]
        sr_info = self._format_sr_check(d)
        self.ersd_status.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.ersd_status.setText(
            f"✓ <b>{os.path.basename(path)}</b> · "
            f"{n_ch} canais · {n_mi} trials MI · "
            f"{n_base} baselines · classes: {cls_str}<br>"
            + sr_info
        )
        self.ersd_compute_btn.setEnabled(True)

    def _ersd_compute(self):
        if not self._ersd_data:
            return
        d = self._ersd_data
        eeg, sr, names = d["eeg"], d["sr"], d["ch_names"]
        trials = d.get("trials") or []
        cls = self.ersd_class_combo.currentData()
        band_name = self.ersd_band_combo.currentText()
        bl_pref = self.ersd_baseline_combo.currentData()
        low, high = self.ERSD_BANDS[band_name]

        # Filtra MI trials
        mi_trials = [t for t in trials if t.get("phase") == "mi"
                     and (cls < 0 or t.get("class_id") == cls)]
        if not mi_trials:
            QtWidgets.QMessageBox.warning(self, "ERS/ERD",
                "Nenhum trial MI dessa classe no events.csv.")
            return

        # Para cada MI trial, encontra a baseline mais próxima ANTES dele
        if bl_pref == "auto":
            bl_phases = self.BASELINE_PHASES
        else:
            bl_phases = (bl_pref,)
        bl_trials = [t for t in trials if t.get("phase") in bl_phases]

        # Para cada MI trial, pega a fase de baseline imediatamente anterior
        # (com end_line < start_line do MI). Se não houver, usa o trial baseline
        # mais próximo (anterior).
        mi_ranges, bl_ranges = [], []
        for mi in mi_trials:
            mi_s, mi_e = mi["start_line"], mi["end_line"]
            best = None
            best_gap = float("inf")
            for b in bl_trials:
                if b["end_line"] <= mi_s:
                    gap = mi_s - b["end_line"]
                    if gap < best_gap:
                        best_gap = gap; best = b
            if best is None and bl_trials:
                best = bl_trials[0]
            if best is None:
                continue
            mi_ranges.append((mi_s, mi_e))
            bl_ranges.append((best["start_line"], best["end_line"]))

        if not mi_ranges:
            QtWidgets.QMessageBox.warning(self, "ERS/ERD",
                "Sem pares MI+baseline emparelháveis.")
            return

        # ERD por canal (média entre trials)
        ersd = SignalProcessor.compute_ersd_per_channel(
            eeg, sr, mi_ranges, bl_ranges, low, high)
        n_ch = len(ersd)

        # Atualiza barras
        self.ersd_bars.setOpts(
            x=list(range(n_ch)), height=ersd.tolist(),
            brushes=[pg.mkBrush("#22dd33" if v > 0 else COLORS["error"])
                     for v in ersd])
        self.ersd_bar_plot.getAxis("bottom").setTicks(
            [list(enumerate(names))])

        # Atualiza head plot (passa valores reais, normalizado interno)
        # O HeadPlot espera valores >= 0; convertemos: usamos |ERD%| e sinal
        # no tooltip via raw_values + powers normalizado
        try:
            # Para mostrar topografia bipolar significativa: invertemos sinal
            # de ERD% (mais negativo = mais escuro/quente) — passamos +|ERD|
            abs_vals = np.abs(ersd[:n_ch])
            # Estende para MAX_CHANNELS preenchendo com 0
            padded = np.zeros(MAX_CHANNELS)
            padded[:n_ch] = abs_vals
            self.ersd_head.set_powers(padded.tolist(),
                                       band_name=f"|ERD%| {band_name.split()[0]}")
        except Exception: pass

        # Curso temporal C3/C4/Cz (índices nesses canais)
        # Calcula potência baseline média por canal para normalização
        n_chs = eeg.shape[0]
        base_pwr = np.zeros(n_chs)
        for ch in range(n_chs):
            ratios = []
            for (bl_s, bl_e) in bl_ranges:
                a = max(0, bl_s - 1); b = min(eeg.shape[1], bl_e)
                if b - a < int(sr * 0.5): continue
                ratios.append(SignalProcessor.compute_band_power(
                    eeg[ch, a:b], low, high, sr))
            base_pwr[ch] = float(np.mean(ratios)) if ratios else 1e-9
        t_axis, ersd_t = SignalProcessor.compute_ersd_timecourse(
            eeg, sr, mi_ranges, base_pwr, low, high)
        # Plota C3, C4, Cz se presentes
        def _ch_idx(label):
            try: return names.index(label)
            except ValueError: return -1
        i_c3 = _ch_idx("C3"); i_c4 = _ch_idx("C4"); i_cz = _ch_idx("Cz")
        if i_c3 >= 0 and t_axis.size:
            self.ersd_c3_curve.setData(t_axis, ersd_t[i_c3])
        else:
            self.ersd_c3_curve.setData([], [])
        if i_c4 >= 0 and t_axis.size:
            self.ersd_c4_curve.setData(t_axis, ersd_t[i_c4])
        else:
            self.ersd_c4_curve.setData([], [])
        if i_cz >= 0 and t_axis.size:
            self.ersd_cz_curve.setData(t_axis, ersd_t[i_cz])
        else:
            self.ersd_cz_curve.setData([], [])

        # Sumário
        cls_name = (self.BCI_CLASS_NAMES.get(cls, "Todas") if cls >= 0 else "Todas")
        most_neg = int(np.argmin(ersd[:n_ch]))
        self.ersd_info_label.setText(
            f"N trials: {len(mi_ranges)} | Classe: {cls_name} | Banda: {band_name}  "
            f"| Maior ERD: {names[most_neg]} ({ersd[most_neg]:+.1f}%) "
            f"| Mediana: {float(np.median(ersd[:n_ch])):+.1f}%"
        )
        self._audit_event("ersd_compute",
                          n_trials=len(mi_ranges), classe=cls_name,
                          band=band_name)

    # ---- Tab: Conexão ----
    def _build_connection_tab(self):
        # Helper local: label com altura minima garantida (evita corte vertical
        # quando a tela e pequena e o QGridLayout tenta espremer demais).
        def _h_label(text, role="form"):
            lbl = QtWidgets.QLabel(text)
            lbl.setMinimumHeight(28)
            if role == "form":
                lbl.setStyleSheet(f"color: {COLORS['text']}; font-weight: bold;")
            return lbl

        # Widget interno (conteudo real) + QScrollArea externa, para que a aba
        # nunca tenha widgets "comidos" verticalmente em telas pequenas.
        inner = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(inner)
        layout.setContentsMargins(20, 20, 20, 20); layout.setSpacing(15)

        conn_group = QtWidgets.QGroupBox("Parâmetros de Conexão")
        grid = QtWidgets.QGridLayout(conn_group)
        grid.setContentsMargins(12, 14, 12, 12)
        grid.setVerticalSpacing(12); grid.setHorizontalSpacing(15)
        # Forca altura minima das linhas — corrige bug "labels cortados"
        # observado em telas baixas (QGridLayout sem hint de altura espreme).
        for r in range(5):
            grid.setRowMinimumHeight(r, 34)

        grid.addWidget(_h_label("Porta COM:"), 0, 0)
        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.setMinimumWidth(280); self.port_combo.setMinimumHeight(30)
        grid.addWidget(self.port_combo, 0, 1)
        self.refresh_btn = QtWidgets.QPushButton("Atualizar")
        self.refresh_btn.setMinimumHeight(30)
        self.refresh_btn.clicked.connect(self._refresh_ports)
        grid.addWidget(self.refresh_btn, 0, 2)

        grid.addWidget(_h_label("Baud Rate:"), 1, 0)
        self.baud_combo = QtWidgets.QComboBox()
        self.baud_combo.setMinimumHeight(30)
        self.baud_combo.addItems(["9600","19200","38400","57600","115200","230400","460800","921600"])
        self.baud_combo.setCurrentText("115200"); self.baud_combo.setEditable(True)
        grid.addWidget(self.baud_combo, 1, 1)

        grid.addWidget(_h_label("Modo:"), 2, 0)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.setMinimumHeight(30)
        self.mode_combo.addItems([
            "Hardware (porta COM real)",
            "Simulação (sinal sintético)",
            "Playback (replay de CSV)",
        ])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        grid.addWidget(self.mode_combo, 2, 1)

        grid.addWidget(_h_label("Arquivo de playback:"), 3, 0)
        pb_row = QtWidgets.QHBoxLayout(); pb_row.setSpacing(6)
        self.playback_path_edit = QtWidgets.QLineEdit()
        self.playback_path_edit.setMinimumHeight(30)
        self.playback_path_edit.setPlaceholderText("(selecione um CSV gravado anteriormente)")
        self.playback_path_edit.setEnabled(False)
        pb_row.addWidget(self.playback_path_edit, stretch=1)
        self.playback_browse_btn = QtWidgets.QPushButton("Procurar...")
        self.playback_browse_btn.setMinimumHeight(30)
        self.playback_browse_btn.setEnabled(False)
        self.playback_browse_btn.clicked.connect(self._select_playback_file)
        pb_row.addWidget(self.playback_browse_btn)
        pb_widget = QtWidgets.QWidget(); pb_widget.setLayout(pb_row)
        grid.addWidget(pb_widget, 3, 1, 1, 2)

        grid.addWidget(_h_label("Diretório de sessões:"), 4, 0)
        self.conn_save_dir_label = QtWidgets.QLabel(self.config.save_directory)
        self.conn_save_dir_label.setMinimumHeight(28)
        self.conn_save_dir_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-style: italic;")
        self.conn_save_dir_label.setWordWrap(True)
        grid.addWidget(self.conn_save_dir_label, 4, 1, 1, 2)
        layout.addWidget(conn_group)

        # ============================================================
        # Grupo BLUETOOTH — scan e listagem de dispositivos BLE
        # ============================================================
        bt_group = QtWidgets.QGroupBox("Bluetooth (BLE) — Dispositivos Pareáveis")
        bt_layout = QtWidgets.QVBoxLayout(bt_group)
        bt_layout.setContentsMargins(10, 12, 10, 10); bt_layout.setSpacing(8)

        bt_info = QtWidgets.QLabel(
            "Dispositivos Bluetooth Low Energy ao redor. "
            "Para placas com perfil <b>Serial Port</b> (HC-05, Cyton-BT, etc.), "
            "primeiro pareie no Windows (Configurações → Bluetooth) — uma porta "
            "COM virtual será criada e aparecerá na lista de portas acima. "
            "Para dispositivos BLE puros (Ganglion, etc.), use o scan abaixo."
        )
        bt_info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        bt_info.setStyleSheet(f"color: {COLORS['text_dim']};")
        bt_info.setWordWrap(True)
        bt_info.setMinimumHeight(70)
        bt_layout.addWidget(bt_info)

        bt_row = QtWidgets.QHBoxLayout(); bt_row.setSpacing(8)
        self.bt_scan_btn = QtWidgets.QPushButton("Procurar dispositivos (8 s)")
        self.bt_scan_btn.setMinimumHeight(32)
        self.bt_scan_btn.clicked.connect(self._bt_scan_start)
        bt_row.addWidget(self.bt_scan_btn)
        self.bt_status_lbl = QtWidgets.QLabel("Pronto.")
        self.bt_status_lbl.setMinimumHeight(28)
        self.bt_status_lbl.setStyleSheet(f"color: {COLORS['text_dim']};")
        bt_row.addWidget(self.bt_status_lbl, stretch=1)
        self.bt_pair_help_btn = QtWidgets.QPushButton("Como parear no Windows")
        self.bt_pair_help_btn.setMinimumHeight(32)
        self.bt_pair_help_btn.clicked.connect(self._bt_show_pair_help)
        bt_row.addWidget(self.bt_pair_help_btn)
        bt_layout.addLayout(bt_row)

        self.bt_devices_table = QtWidgets.QTableWidget(0, 4)
        self.bt_devices_table.setHorizontalHeaderLabels(
            ["Nome", "Endereço", "RSSI (dBm)", "Tipo"])
        self.bt_devices_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.bt_devices_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.bt_devices_table.verticalHeader().setVisible(False)
        bt_hdr = self.bt_devices_table.horizontalHeader()
        bt_hdr.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        bt_hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        # Altura mínima/máxima ampliadas para evitar tabela "comida".
        # Em telas pequenas a scroll area da aba toma conta.
        self.bt_devices_table.setMinimumHeight(160)
        self.bt_devices_table.setMaximumHeight(260)
        bt_layout.addWidget(self.bt_devices_table)

        # Estado do scanner
        self._bt_scanner_thread = None

        layout.addWidget(bt_group)

        # ============================================================
        # Grupo de EXPANSAO — seletor multi-step de canais ativos + auto-detecção
        # ============================================================
        exp_group = QtWidgets.QGroupBox("Expansão de Canais")
        exp_layout = QtWidgets.QVBoxLayout(exp_group)
        exp_layout.setContentsMargins(10, 10, 10, 10); exp_layout.setSpacing(8)
        exp_row = QtWidgets.QHBoxLayout(); exp_row.setSpacing(10)

        exp_lbl = QtWidgets.QLabel("Total de canais ativos:")
        exp_lbl.setStyleSheet(
            f"color: {COLORS['expansion']}; font-weight: bold; font-size: 11pt;")
        exp_lbl.setMinimumHeight(28)
        exp_row.addWidget(exp_lbl)

        self.expansion_combo = QtWidgets.QComboBox()
        self.expansion_combo.setMinimumWidth(260)
        self.expansion_combo.setMinimumHeight(30)
        for n in EXPANSION_STEPS:
            extra = n - BASE_CHANNELS
            if n == BASE_CHANNELS:
                self.expansion_combo.addItem(f"{n} canais — placa base", n)
            elif n <= CYTON_MAX_CHANNELS:
                self.expansion_combo.addItem(
                    f"{n} canais — base + módulo Daisy (+{extra})", n)
            else:
                self.expansion_combo.addItem(
                    f"{n} canais — base + {extra//8} módulos de expansão "
                    f"(placa customizada)", n)
        self.expansion_combo.setCurrentIndex(0)  # default: 8 canais
        self.expansion_combo.currentIndexChanged.connect(self._on_expansion_combo_changed)
        exp_row.addWidget(self.expansion_combo)
        exp_row.addStretch()
        detect_btn = QtWidgets.QPushButton("Detectar expansão automaticamente")
        detect_btn.setMinimumHeight(30)
        detect_btn.clicked.connect(self._detect_expansion)
        exp_row.addWidget(detect_btn)
        exp_layout.addLayout(exp_row)

        # Alias retrocompativel: muitos handlers usam self.expansion_toggle
        # como QCheckBox binario. Mantemos o checkbox oculto, sincronizado com
        # o combo (>= 16ch -> checked), de modo que codigo legado nao quebre.
        self.expansion_toggle = QtWidgets.QCheckBox()
        self.expansion_toggle.setVisible(False)
        self.expansion_toggle.toggled.connect(self._on_expansion_toggled)

        exp_info = QtWidgets.QLabel(
            "<b>Modos disponíveis:</b><br>"
            f"&nbsp;&nbsp;• <b>8 canais</b>: apenas a placa base (Cyton sozinho — 250 Hz nativo).<br>"
            f"&nbsp;&nbsp;• <b>16 canais</b>: placa base + módulo Daisy (Cyton+Daisy — 125 Hz efetivo).<br>"
            f"&nbsp;&nbsp;• <b>24–64 canais</b>: placa customizada Bionica Lab com expansão multi-bloco "
            f"(EEG / EMG / ECG / EoG multimodal, taxa configurável).<br>"
            f"<b>Detecção automática</b>: em <i>Playback</i>, conta as colunas do CSV e ajusta o modo; "
            f"em <i>Hardware</i>, envia comando '?' ao dispositivo.<br>"
            f"<b>Canais base (CH1-{BASE_CHANNELS})</b>: cor padrão da paleta. "
            f"<b>Canais de expansão (CH{BASE_CHANNELS+1}-{MAX_CHANNELS})</b>: borda azulada "
            f"para distinção visual."
        )
        exp_info.setStyleSheet(f"color: {COLORS['text_dim']};")
        exp_info.setWordWrap(True)
        exp_info.setMinimumHeight(110)
        exp_layout.addWidget(exp_info)
        layout.addWidget(exp_group)

        # Botões
        btn_row = QtWidgets.QHBoxLayout()
        self.connect_btn = QtWidgets.QPushButton("▶  Conectar")
        self.connect_btn.setMinimumHeight(46)
        self.connect_btn.clicked.connect(self._toggle_connection)
        btn_row.addWidget(self.connect_btn)
        self.record_btn = QtWidgets.QPushButton("●  Iniciar Gravação")
        self.record_btn.setMinimumHeight(46); self.record_btn.setEnabled(False)
        self.record_btn.clicked.connect(self._toggle_recording)
        btn_row.addWidget(self.record_btn)
        # Demo 30s — protocolo automático
        self.demo_btn = QtWidgets.QPushButton("Demo 30s")
        self.demo_btn.setMinimumHeight(46)
        self.demo_btn.setToolTip(
            "Protocolo automático: simulação 30s alternando 'olhos_abertos'/"
            "'olhos_fechados' a cada 5s. Grava CSV + gera PDF ao final."
        )
        self.demo_btn.clicked.connect(self._start_demo_mode)
        btn_row.addWidget(self.demo_btn)
        layout.addLayout(btn_row)

        self.playback_progress = QtWidgets.QProgressBar()
        self.playback_progress.setRange(0, 1000); self.playback_progress.setVisible(False)
        layout.addWidget(self.playback_progress)

        log_group = QtWidgets.QGroupBox("Log da Sessão")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self.log_view = QtWidgets.QTextEdit(); self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(120)
        log_layout.addWidget(self.log_view)
        layout.addWidget(log_group, stretch=1)

        # Envolve toda a aba em QScrollArea: garante que nenhum grupo seja
        # "comido" verticalmente em telas pequenas (corrige bug de labels
        # cortados em Parâmetros de Conexão, Bluetooth e Expansão).
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(inner)
        return scroll

    # ==================================================================
    # Focus / SSVEP — índice de foco/concentração baseado em bandas EEG
    # ==================================================================
    def _build_focus_tab(self):
        """Aba Focus / SSVEP — métrica de foco em tempo real.

        Calcula 2 razões clássicas a partir do PSD do canal selecionado:
            Engagement  = Beta / (Alpha + Theta)         (escala 0..2+)
            Calmness    = Alpha / Beta                    (escala 0..3+)
            SSVEP power = banda 1Hz em torno da freq alvo
        E mantém um histórico temporal das 3 métricas.
        """
        widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(8, 8, 8, 8); outer.setSpacing(6)

        # === Controles ===
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Canal EEG:"))
        self.focus_channel_combo = QtWidgets.QComboBox()
        for ch in range(MAX_CHANNELS):
            elec = self.config.channel_mapping[ch] if ch < len(self.config.channel_mapping) else f"E{ch+1}"
            self.focus_channel_combo.addItem(f"CH{ch+1} ({elec})", ch)
        # Default: tenta achar Oz / Cz / Fz para foco/SSVEP
        for target in ("Oz", "Cz", "Pz", "Fz"):
            idx_pref = next((i for i in range(MAX_CHANNELS)
                             if i < len(self.config.channel_mapping)
                             and self.config.channel_mapping[i] == target), None)
            if idx_pref is not None:
                self.focus_channel_combo.setCurrentIndex(idx_pref)
                break
        ctrl.addWidget(self.focus_channel_combo)
        ctrl.addSpacing(15)
        ctrl.addWidget(QtWidgets.QLabel("Freq SSVEP alvo (Hz):"))
        self.focus_ssvep_freq = QtWidgets.QDoubleSpinBox()
        self.focus_ssvep_freq.setRange(4.0, 50.0)
        self.focus_ssvep_freq.setValue(12.0)
        self.focus_ssvep_freq.setSingleStep(1.0)
        self.focus_ssvep_freq.setSuffix(" Hz")
        self.focus_ssvep_freq.setToolTip(
            "Frequência do estímulo visual flicker (LED/tela). "
            "Comum: 7.5, 8.57, 10, 12, 15 Hz."
        )
        ctrl.addWidget(self.focus_ssvep_freq)
        ctrl.addStretch()
        btn_baseline = QtWidgets.QPushButton("Definir baseline (5s)")
        btn_baseline.setToolTip("Coleta 5s de baseline para normalizar o índice de foco.")
        btn_baseline.clicked.connect(self._focus_start_baseline)
        ctrl.addWidget(btn_baseline)
        btn_reset = QtWidgets.QPushButton("Reset")
        btn_reset.clicked.connect(self._focus_reset)
        ctrl.addWidget(btn_reset)
        outer.addLayout(ctrl)

        # === Cards de métricas ===
        cards = QtWidgets.QHBoxLayout()
        for key, label, color in (
            ("engagement", "Engagement",  COLORS["accent"]),
            ("calmness",   "Calmness",    "#a3ff66"),
            ("ssvep",      "SSVEP Power", "#eebb33"),
            ("state",      "Estado",      COLORS["warning"]),
        ):
            box = QtWidgets.QVBoxLayout()
            val = QtWidgets.QLabel("--")
            val.setStyleSheet(
                f"color: {color}; font-size: 22pt; font-weight: bold; "
                f"font-family: {FONT_DATA_STACK}; padding: 4px 6px;")
            val.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            val.setMinimumHeight(50)
            box.addWidget(val)
            lbl = QtWidgets.QLabel(label)
            lbl.setStyleSheet(f"color: {COLORS['text_dim']};")
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            box.addWidget(lbl)
            if not hasattr(self, "_focus_metric_lbls"):
                self._focus_metric_lbls = {}
            self._focus_metric_lbls[key] = val
            cards.addLayout(box)
            cards.addSpacing(20)
        cards.addStretch()
        outer.addLayout(cards)

        # === FFT + Histórico temporal ===
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        # FFT do canal
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left); ll.setContentsMargins(2, 2, 2, 2)
        ll.addWidget(QtWidgets.QLabel("Espectro do canal selecionado"))
        self.focus_fft_plot = pg.PlotWidget(enableMenu=False)
        self.focus_fft_plot.showGrid(x=True, y=True, alpha=0.15)
        self.focus_fft_plot.setLabel("left",   "Amplitude", units="µV")
        self.focus_fft_plot.setLabel("bottom", "Freq",      units="Hz")
        self.focus_fft_plot.setXRange(0, 40)
        self.focus_fft_plot.setMenuEnabled(False)
        self.focus_fft_curve = self.focus_fft_plot.plot(
            pen=pg.mkPen(COLORS["accent"], width=1.4))
        # Marca bandas Mu/Beta com linhas
        for _, (lo, _hi) in EEG_BANDS.items():
            self.focus_fft_plot.addItem(pg.InfiniteLine(
                pos=lo, angle=90,
                pen=pg.mkPen(COLORS["border"], style=QtCore.Qt.PenStyle.DashLine)))
        # Linha vertical em freq SSVEP alvo
        self.focus_ssvep_line = pg.InfiniteLine(
            pos=12.0, angle=90,
            pen=pg.mkPen("#eebb33", width=2, style=QtCore.Qt.PenStyle.DotLine))
        self.focus_fft_plot.addItem(self.focus_ssvep_line)
        self.focus_ssvep_freq.valueChanged.connect(
            lambda v: self.focus_ssvep_line.setPos(v))
        ll.addWidget(self.focus_fft_plot)
        split.addWidget(left)
        # Histórico temporal
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right); rl.setContentsMargins(2, 2, 2, 2)
        rl.addWidget(QtWidgets.QLabel("Histórico (últimos ~60 s)"))
        self.focus_time_plot = pg.PlotWidget(enableMenu=False)
        self.focus_time_plot.showGrid(x=True, y=True, alpha=0.15)
        self.focus_time_plot.setLabel("left", "Métrica (normalizada)")
        self.focus_time_plot.setLabel("bottom", "Tempo", units="s")
        self.focus_time_plot.setMenuEnabled(False)
        self.focus_time_plot.addLegend(offset=(10, 10))
        self.focus_eng_curve = self.focus_time_plot.plot(
            pen=pg.mkPen(COLORS["accent"], width=1.6), name="Engagement")
        self.focus_calm_curve = self.focus_time_plot.plot(
            pen=pg.mkPen("#a3ff66", width=1.6), name="Calmness")
        self.focus_ssvep_curve = self.focus_time_plot.plot(
            pen=pg.mkPen("#eebb33", width=1.6), name="SSVEP")
        rl.addWidget(self.focus_time_plot)
        split.addWidget(right)
        split.setSizes([520, 600])
        outer.addWidget(split, stretch=1)

        # Estado interno
        self._focus_history = {"t": [], "engagement": [], "calmness": [], "ssvep": []}
        self._focus_baseline = {"engagement": None, "calmness": None, "ssvep": None}
        self._focus_baseline_collecting = False
        self._focus_baseline_data = []
        self._focus_baseline_start_t = 0.0
        return widget

    def _focus_reset(self):
        self._focus_history = {"t": [], "engagement": [], "calmness": [], "ssvep": []}
        self._focus_baseline = {"engagement": None, "calmness": None, "ssvep": None}
        if hasattr(self, "focus_eng_curve"):
            self.focus_eng_curve.setData([], [])
            self.focus_calm_curve.setData([], [])
            self.focus_ssvep_curve.setData([], [])
        for k in ("engagement", "calmness", "ssvep", "state"):
            if k in self._focus_metric_lbls:
                self._focus_metric_lbls[k].setText("--")

    def _focus_start_baseline(self):
        """Inicia coleta de baseline 5s para normalizar métricas."""
        self._focus_baseline_collecting = True
        self._focus_baseline_data = []
        self._focus_baseline_start_t = time.time()
        self._log("Coletando baseline de foco (5s — relaxe e olhe um ponto fixo)...")

    def _compute_focus_metrics(self, signal, fs, ssvep_freq):
        """Calcula engagement, calmness e SSVEP power do canal."""
        if len(signal) < int(fs * 1.0):
            return None
        # PSD via FFT
        try:
            n = len(signal)
            # Remove DC + linear trend
            sig = signal - np.mean(signal)
            window = np.hanning(n)
            spec = np.fft.rfft(sig * window)
            freqs = np.fft.rfftfreq(n, 1.0 / fs)
            psd = (np.abs(spec) ** 2) / (n * fs)
        except Exception:
            return None
        # Bandas
        def band_power(f_lo, f_hi):
            mask = (freqs >= f_lo) & (freqs < f_hi)
            return float(_TRAPEZOID(psd[mask], freqs[mask])) if mask.any() else 0.0
        theta = band_power(4, 8)
        alpha = band_power(8, 13)
        beta  = band_power(13, 30)
        # Engagement = Beta / (Alpha + Theta) — Pope et al. 1995
        eng = beta / (alpha + theta) if (alpha + theta) > 0 else 0.0
        calm = alpha / beta if beta > 0 else 0.0
        # SSVEP: pico em ±0.5 Hz da freq alvo, normalizado pela vizinhança
        mask_s = (freqs >= ssvep_freq - 0.5) & (freqs <= ssvep_freq + 0.5)
        mask_n = ((freqs >= ssvep_freq - 2.0) & (freqs < ssvep_freq - 0.5)) | \
                 ((freqs > ssvep_freq + 0.5) & (freqs <= ssvep_freq + 2.0))
        sig_p = float(np.mean(psd[mask_s])) if mask_s.any() else 0.0
        noi_p = float(np.mean(psd[mask_n])) if mask_n.any() else 1e-9
        ssvep_snr = sig_p / max(noi_p, 1e-9)
        return {
            "engagement": eng,
            "calmness":   calm,
            "ssvep":      ssvep_snr,
            "freqs": freqs, "psd": psd,
        }

    def _update_focus_view(self):
        """Atualiza FFT, métricas e histórico de foco."""
        if not hasattr(self, "focus_fft_plot"): return
        ch = self.focus_channel_combo.currentData() if hasattr(self, "focus_channel_combo") else 0
        if ch is None or ch < 0: ch = 0
        data = self._ordered_buffer()
        if data.shape[1] < int(SAMPLE_RATE * 1.0): return
        sig = data[ch].astype(np.float64)
        # Pega último ~2s para latência menor
        n_win = min(len(sig), int(SAMPLE_RATE * 2.0))
        win = sig[-n_win:]
        m = self._compute_focus_metrics(win, SAMPLE_RATE,
                                         self.focus_ssvep_freq.value())
        if m is None: return
        # FFT plot
        self.focus_fft_curve.setData(m["freqs"], np.sqrt(m["psd"]))
        # Baseline em coleta
        if self._focus_baseline_collecting:
            self._focus_baseline_data.append((m["engagement"], m["calmness"], m["ssvep"]))
            if (time.time() - self._focus_baseline_start_t) > 5.0:
                # Finaliza baseline com média
                if self._focus_baseline_data:
                    arr = np.array(self._focus_baseline_data)
                    self._focus_baseline = {
                        "engagement": float(np.mean(arr[:, 0])),
                        "calmness":   float(np.mean(arr[:, 1])),
                        "ssvep":      float(np.mean(arr[:, 2])),
                    }
                self._focus_baseline_collecting = False
                self._log(f"Baseline definido: Eng={self._focus_baseline['engagement']:.2f} "
                          f"Calm={self._focus_baseline['calmness']:.2f}")
        # Atualiza cards
        def show(key, fmt):
            v = m[key]
            base = self._focus_baseline.get(key)
            if base is not None and base > 0:
                v_norm = v / base
                self._focus_metric_lbls[key].setText(f"{v_norm:.2f}×")
            else:
                self._focus_metric_lbls[key].setText(fmt.format(v))
        show("engagement", "{:.2f}")
        show("calmness",   "{:.2f}")
        show("ssvep",      "{:.2f}")
        # Estado discreto
        if self._focus_baseline.get("engagement"):
            ratio = m["engagement"] / self._focus_baseline["engagement"]
            if   ratio > 1.4: state = "FOCADO"; color = COLORS["warning"]
            elif ratio < 0.8: state = "RELAXADO"; color = "#a3ff66"
            else:             state = "NORMAL"; color = COLORS["text"]
        else:
            state = "(sem baseline)"; color = COLORS["text_dim"]
        self._focus_metric_lbls["state"].setText(state)
        self._focus_metric_lbls["state"].setStyleSheet(
            f"color: {color}; font-size: 22pt; font-weight: bold;")
        # Histórico
        now = time.time()
        H = self._focus_history
        H["t"].append(now)
        H["engagement"].append(m["engagement"])
        H["calmness"].append(m["calmness"])
        H["ssvep"].append(m["ssvep"])
        # Mantém últimos 120 pontos (~60s a 2Hz)
        for k in ("t", "engagement", "calmness", "ssvep"):
            if len(H[k]) > 120:
                H[k] = H[k][-120:]
        t0 = H["t"][0] if H["t"] else now
        t_rel = [t - t0 for t in H["t"]]
        # Normaliza com baseline (ou bruto)
        def norm(arr, key):
            base = self._focus_baseline.get(key)
            if base and base > 0:
                return [v / base for v in arr]
            return arr
        self.focus_eng_curve.setData(t_rel,  norm(H["engagement"], "engagement"))
        self.focus_calm_curve.setData(t_rel, norm(H["calmness"],   "calmness"))
        self.focus_ssvep_curve.setData(t_rel, norm(H["ssvep"],     "ssvep"))

    # ==================================================================
    # EMG Joystick — mapeamento de 4 canais EMG para X/Y de joystick virtual
    # ==================================================================
    # ==================================================================
    # BCI Trainer (Motor Imagery — CSP + LDA simplificados)
    # ==================================================================
    def _build_bci_trainer_tab(self):
        """Treinador BCI de Imaginação Motora (Esquerda vs Direita).

        Pipeline:
            1. Calibração: N trials por classe (8s cada: 2s preparação + 4s MI + 2s descanso)
            2. Treino: CSP (6 filtros) + LDA binário
            3. Online: classifica em tempo real, mostra barra de feedback

        Sem dependência de scikit-learn: CSP via scipy eigh + LDA Fisher manual.
        """
        widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(8, 8, 8, 8); outer.setSpacing(8)

        info = QtWidgets.QLabel(
            "<b>BCI Trainer — Imaginação Motora (MI)</b><br>"
            "Treina classificador <b>CSP+LDA</b> para discriminar 2 classes "
            "(Esquerda vs Direita). Use canais centrais — <b>C3</b> (esquerda) e "
            "<b>C4</b> (direita) — durante a calibração."
        )
        info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        info.setStyleSheet(f"color: {COLORS['text_dim']}; padding: 4px;")
        info.setWordWrap(True)
        outer.addWidget(info)

        # ---- Controles ----
        ctrl_group = QtWidgets.QGroupBox("Parâmetros")
        cgl = QtWidgets.QGridLayout(ctrl_group)
        cgl.addWidget(QtWidgets.QLabel("Trials por classe:"), 0, 0)
        self.bci_trials_spin = QtWidgets.QSpinBox()
        self.bci_trials_spin.setRange(5, 100); self.bci_trials_spin.setValue(20)
        cgl.addWidget(self.bci_trials_spin, 0, 1)
        cgl.addWidget(QtWidgets.QLabel("Duração MI (s):"), 0, 2)
        self.bci_dur_spin = QtWidgets.QDoubleSpinBox()
        self.bci_dur_spin.setRange(1.0, 10.0); self.bci_dur_spin.setValue(4.0)
        self.bci_dur_spin.setSingleStep(0.5)
        cgl.addWidget(self.bci_dur_spin, 0, 3)
        cgl.addWidget(QtWidgets.QLabel("Banda Mu/Beta (Hz):"), 1, 0)
        self.bci_band_lo = QtWidgets.QDoubleSpinBox()
        self.bci_band_lo.setRange(4.0, 25.0); self.bci_band_lo.setValue(8.0)
        cgl.addWidget(self.bci_band_lo, 1, 1)
        self.bci_band_hi = QtWidgets.QDoubleSpinBox()
        self.bci_band_hi.setRange(10.0, 50.0); self.bci_band_hi.setValue(30.0)
        cgl.addWidget(self.bci_band_hi, 1, 2)
        cgl.addWidget(QtWidgets.QLabel("Canais (1-based, csv):"), 2, 0)
        self.bci_channels_edit = QtWidgets.QLineEdit("3,4")  # default C3/C4
        self.bci_channels_edit.setToolTip("Lista de canais usados no CSP. Ex.: 3,4 = CH3 e CH4.")
        cgl.addWidget(self.bci_channels_edit, 2, 1, 1, 3)
        outer.addWidget(ctrl_group)

        # ---- Status + barras ----
        status_group = QtWidgets.QGroupBox("Estado / Feedback")
        sgl = QtWidgets.QVBoxLayout(status_group)
        self.bci_status_lbl = QtWidgets.QLabel("Pronto. Clique 'Calibrar' para começar.")
        self.bci_status_lbl.setStyleSheet(
            f"color: {COLORS['text']}; font-size: 12pt; padding: 6px;")
        self.bci_status_lbl.setMinimumHeight(40)
        sgl.addWidget(self.bci_status_lbl)
        # Barra de progresso da calibração
        self.bci_calib_progress = QtWidgets.QProgressBar()
        self.bci_calib_progress.setRange(0, 100)
        self.bci_calib_progress.setValue(0)
        self.bci_calib_progress.setFormat("Calibração: %p%")
        sgl.addWidget(self.bci_calib_progress)
        # Feedback online (barra horizontal -1 a +1, esq vs dir)
        self.bci_feedback_bar = QtWidgets.QProgressBar()
        self.bci_feedback_bar.setRange(-100, 100)
        self.bci_feedback_bar.setValue(0)
        self.bci_feedback_bar.setFormat("← Esquerda  |  Direita →")
        self.bci_feedback_bar.setStyleSheet(
            f"QProgressBar {{ border: 1px solid {COLORS['border']}; "
            f"text-align: center; height: 30px; }} "
            f"QProgressBar::chunk {{ background-color: {COLORS['accent']}; }}")
        sgl.addWidget(self.bci_feedback_bar)
        # Acurácia
        self.bci_accuracy_lbl = QtWidgets.QLabel("Acurácia (treino): --")
        self.bci_accuracy_lbl.setStyleSheet(
            f"color: {SIGNAL_TYPE_COLORS['EEG']}; font-weight: bold;")
        sgl.addWidget(self.bci_accuracy_lbl)
        outer.addWidget(status_group)

        # ---- Botões de ação ----
        btn_row = QtWidgets.QHBoxLayout()
        self.bci_calib_left_btn = QtWidgets.QPushButton("Calibrar ESQUERDA (1 trial)")
        self.bci_calib_left_btn.clicked.connect(lambda: self._bci_record_trial(0))
        btn_row.addWidget(self.bci_calib_left_btn)
        self.bci_calib_right_btn = QtWidgets.QPushButton("Calibrar DIREITA (1 trial)")
        self.bci_calib_right_btn.clicked.connect(lambda: self._bci_record_trial(1))
        btn_row.addWidget(self.bci_calib_right_btn)
        self.bci_train_btn = QtWidgets.QPushButton("Treinar CSP+LDA")
        self.bci_train_btn.clicked.connect(self._bci_train_classifier)
        btn_row.addWidget(self.bci_train_btn)
        self.bci_online_btn = QtWidgets.QPushButton("Iniciar Online")
        self.bci_online_btn.setCheckable(True)
        self.bci_online_btn.clicked.connect(self._bci_toggle_online)
        self.bci_online_btn.setEnabled(False)
        btn_row.addWidget(self.bci_online_btn)
        self.bci_reset_btn = QtWidgets.QPushButton("Reset")
        self.bci_reset_btn.clicked.connect(self._bci_reset)
        btn_row.addWidget(self.bci_reset_btn)
        outer.addLayout(btn_row)

        # Estado interno
        self._bci_trials = {0: [], 1: []}   # 0=esq, 1=dir → lista de janelas (n_ch, n_samples)
        self._bci_csp_W = None              # filtros CSP (n_filters, n_ch)
        self._bci_lda_w = None              # vetor LDA
        self._bci_lda_threshold = 0.0
        self._bci_online_timer = None
        return widget

    def _bci_parse_channels(self):
        """Parse self.bci_channels_edit -> lista de índices 0-based."""
        chs = []
        try:
            for tok in self.bci_channels_edit.text().replace(";", ",").split(","):
                tok = tok.strip()
                if tok.isdigit():
                    v = int(tok) - 1
                    if 0 <= v < MAX_CHANNELS: chs.append(v)
        except Exception: pass
        return chs

    def _bci_record_trial(self, label):
        """Captura janela MI da última X segundos como trial."""
        chs = self._bci_parse_channels()
        if not chs:
            QtWidgets.QMessageBox.information(
                self, "Canais", "Informe canais (ex.: 3,4)."); return
        data = self._ordered_buffer()
        dur = self.bci_dur_spin.value()
        n_w = int(dur * SAMPLE_RATE)
        if data.shape[1] < n_w:
            QtWidgets.QMessageBox.information(
                self, "Buffer insuficiente",
                f"Aguarde {dur}s de sinal acumulado primeiro."); return
        win = data[chs, -n_w:].astype(np.float64)
        # Bandpass via filtro Butterworth ordem 4
        try:
            sos = scipy_signal.butter(4,
                [self.bci_band_lo.value() / (SAMPLE_RATE/2),
                 self.bci_band_hi.value() / (SAMPLE_RATE/2)],
                btype="bandpass", output="sos")
            win = scipy_signal.sosfiltfilt(sos, win, axis=1)
        except Exception: pass
        self._bci_trials[label].append(win)
        name = "ESQUERDA" if label == 0 else "DIREITA"
        n0 = len(self._bci_trials[0]); n1 = len(self._bci_trials[1])
        target = self.bci_trials_spin.value()
        self.bci_calib_progress.setValue(
            int(100 * min(1.0, (n0 + n1) / (2 * target))))
        self.bci_status_lbl.setText(
            f"Trial {name} #{n0 if label==0 else n1} salvo. "
            f"Total: {n0}/{target} esq, {n1}/{target} dir.")
        self._log(f"BCI trial gravado: {name} (n={n0 if label==0 else n1})")

    def _bci_train_classifier(self):
        """Treina CSP+LDA a partir dos trials gravados."""
        n0 = len(self._bci_trials[0]); n1 = len(self._bci_trials[1])
        if n0 < 5 or n1 < 5:
            QtWidgets.QMessageBox.information(
                self, "Trials insuficientes",
                f"Mínimo 5 trials por classe. Atual: esq={n0}, dir={n1}."); return
        try:
            # Calcula covariância média por classe
            def cov(X): return X @ X.T / X.shape[1]
            S0 = np.mean([cov(t) for t in self._bci_trials[0]], axis=0)
            S1 = np.mean([cov(t) for t in self._bci_trials[1]], axis=0)
            Sc = S0 + S1 + 1e-6 * np.eye(S0.shape[0])
            # CSP: generalized eigenproblem S0 W = λ Sc W
            from scipy.linalg import eigh
            eigvals, eigvecs = eigh(S0, Sc)
            # Pega filtros extremos (mais discriminativos)
            n_ch = S0.shape[0]
            n_filt = min(6, n_ch)
            idxs = list(range(n_filt // 2)) + list(range(-n_filt // 2, 0))
            W = eigvecs[:, idxs].T  # (n_filters, n_ch)
            self._bci_csp_W = W
            # Features: log-variance dos sinais filtrados por CSP
            def features(trial):
                Y = W @ trial
                v = np.var(Y, axis=1)
                return np.log(v / np.sum(v))
            X0 = np.array([features(t) for t in self._bci_trials[0]])
            X1 = np.array([features(t) for t in self._bci_trials[1]])
            # LDA Fisher
            mu0 = X0.mean(0); mu1 = X1.mean(0)
            Sw = np.cov(X0.T) + np.cov(X1.T)
            w = np.linalg.pinv(Sw + 1e-6 * np.eye(Sw.shape[0])) @ (mu1 - mu0)
            self._bci_lda_w = w
            self._bci_lda_threshold = float(0.5 * w @ (mu0 + mu1))
            # Acurácia de treino
            preds0 = (X0 @ w > self._bci_lda_threshold).astype(int)
            preds1 = (X1 @ w > self._bci_lda_threshold).astype(int)
            acc = (np.sum(preds0 == 0) + np.sum(preds1 == 1)) / (n0 + n1)
            self.bci_accuracy_lbl.setText(f"Acurácia (treino): {acc*100:.1f}%")
            self.bci_status_lbl.setText(
                f"Classificador treinado! Acurácia: {acc*100:.1f}%. "
                f"Clique 'Iniciar Online' para feedback em tempo real."
            )
            self.bci_online_btn.setEnabled(True)
            self._audit_event("bci_classifier_trained",
                              n_trials=(n0, n1), accuracy=float(acc))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Falha no treino", f"Erro: {exc}")
            self._log(f"BCI training failed: {exc}", error=True)

    def _bci_toggle_online(self, checked):
        """Inicia/para classificação online com feedback."""
        if not checked:
            if self._bci_online_timer:
                self._bci_online_timer.stop()
                self._bci_online_timer = None
            self.bci_online_btn.setText("Iniciar Online")
            self.bci_status_lbl.setText("Online: parado.")
            return
        if self._bci_csp_W is None:
            self.bci_online_btn.setChecked(False)
            QtWidgets.QMessageBox.information(
                self, "Sem classificador", "Treine o classificador primeiro."); return
        self._bci_online_timer = QTimer(self)
        self._bci_online_timer.timeout.connect(self._bci_online_step)
        self._bci_online_timer.start(250)
        self.bci_online_btn.setText("Parar Online")
        self.bci_status_lbl.setText("Online: classificando MI a cada 250ms...")
        self._audit_event("bci_online_started")

    def _bci_online_step(self):
        """Classifica janela atual e atualiza barra de feedback."""
        if self._bci_csp_W is None or self._bci_lda_w is None: return
        chs = self._bci_parse_channels()
        if not chs: return
        data = self._ordered_buffer()
        n_w = int(self.bci_dur_spin.value() * SAMPLE_RATE)
        if data.shape[1] < n_w: return
        try:
            win = data[chs, -n_w:].astype(np.float64)
            sos = scipy_signal.butter(4,
                [self.bci_band_lo.value() / (SAMPLE_RATE/2),
                 self.bci_band_hi.value() / (SAMPLE_RATE/2)],
                btype="bandpass", output="sos")
            win = scipy_signal.sosfiltfilt(sos, win, axis=1)
            Y = self._bci_csp_W @ win
            v = np.var(Y, axis=1)
            feat = np.log(v / np.sum(v))
            score = float(feat @ self._bci_lda_w - self._bci_lda_threshold)
            # Mapeia para -100..+100 com tanh
            val = int(100 * np.tanh(score))
            self.bci_feedback_bar.setValue(val)
        except Exception: pass

    def _bci_reset(self):
        self._bci_trials = {0: [], 1: []}
        self._bci_csp_W = None
        self._bci_lda_w = None
        self._bci_lda_threshold = 0.0
        if self._bci_online_timer:
            self._bci_online_timer.stop()
            self._bci_online_timer = None
        self.bci_online_btn.setChecked(False)
        self.bci_online_btn.setEnabled(False)
        self.bci_online_btn.setText("Iniciar Online")
        self.bci_calib_progress.setValue(0)
        self.bci_feedback_bar.setValue(0)
        self.bci_accuracy_lbl.setText("Acurácia (treino): --")
        self.bci_status_lbl.setText("Reset. Pronto para nova calibração.")

    def _build_emg_joystick_tab(self):
        """Aba EMG Joystick — controle proporcional via EMG.

        Mapeia 4 canais EMG para 4 direções (X+, X-, Y+, Y-).
        Calibração por canal: max EMG durante contração → normaliza saída [0,1].
        Dead zone evita drift quando músculos relaxados.
        Saída pode ser enviada via UDP (joystick virtual no PC alvo).
        """
        widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(8, 8, 8, 8); outer.setSpacing(6)

        # === Mapeamento ===
        map_group = QtWidgets.QGroupBox("Mapeamento dos 4 canais EMG → eixos")
        mg = QtWidgets.QGridLayout(map_group)
        mg.addWidget(QtWidgets.QLabel("Direção"),       0, 0)
        mg.addWidget(QtWidgets.QLabel("Canal EMG"),     0, 1)
        mg.addWidget(QtWidgets.QLabel("Envelope atual"),0, 2)
        mg.addWidget(QtWidgets.QLabel("Max calibrado"), 0, 3)
        mg.addWidget(QtWidgets.QLabel("Calibrar (3s contração)"), 0, 4)
        self._joy_axes = {}  # {direction: dict}
        directions = [
            ("X+ (direita)", "+x"),
            ("X- (esquerda)", "-x"),
            ("Y+ (cima)",    "+y"),
            ("Y- (baixo)",   "-y"),
        ]
        for row, (label, code) in enumerate(directions, start=1):
            dl = QtWidgets.QLabel(label)
            dl.setStyleSheet(f"color: {SIGNAL_TYPE_COLORS['EMG']}; font-weight: bold;")
            mg.addWidget(dl, row, 0)
            # Combo de canal EMG
            cb = QtWidgets.QComboBox()
            cb.setMinimumWidth(150)
            mg.addWidget(cb, row, 1)
            # Envelope atual
            env_lbl = QtWidgets.QLabel("0.0")
            env_lbl.setStyleSheet(f"color: {COLORS['text']}; font-family: {FONT_DATA_STACK};")
            env_lbl.setMinimumWidth(60)
            mg.addWidget(env_lbl, row, 2)
            # Max calibrado
            max_lbl = QtWidgets.QLabel("--")
            max_lbl.setStyleSheet(f"color: {COLORS['text_dim']}; font-family: {FONT_DATA_STACK};")
            max_lbl.setMinimumWidth(60)
            mg.addWidget(max_lbl, row, 3)
            # Botão calibrar
            btn = QtWidgets.QPushButton("Calibrar")
            btn.clicked.connect(lambda _ck, c=code: self._joy_calibrate(c))
            mg.addWidget(btn, row, 4)
            self._joy_axes[code] = {
                "combo": cb, "env_lbl": env_lbl, "max_lbl": max_lbl,
                "max_value": 100.0, "calibrating": False,
                "calibration_data": [], "calibration_start": 0.0,
            }
        outer.addWidget(map_group)
        self._joy_repopulate_combos()

        # === Configurações ===
        cfg_group = QtWidgets.QGroupBox("Configurações")
        cl = QtWidgets.QHBoxLayout(cfg_group)
        cl.addWidget(QtWidgets.QLabel("Dead zone:"))
        self.joy_dead_zone_spin = QtWidgets.QDoubleSpinBox()
        self.joy_dead_zone_spin.setRange(0.0, 0.5)
        self.joy_dead_zone_spin.setValue(0.15)
        self.joy_dead_zone_spin.setSingleStep(0.05)
        self.joy_dead_zone_spin.setDecimals(2)
        cl.addWidget(self.joy_dead_zone_spin)
        cl.addSpacing(15)
        cl.addWidget(QtWidgets.QLabel("Smoothing (frames):"))
        self.joy_smoothing_spin = QtWidgets.QSpinBox()
        self.joy_smoothing_spin.setRange(1, 20)
        self.joy_smoothing_spin.setValue(5)
        cl.addWidget(self.joy_smoothing_spin)
        cl.addStretch()
        # Output UDP
        self.joy_udp_check = QtWidgets.QCheckBox("Enviar via UDP (\"joy\" namespace)")
        self.joy_udp_check.setToolTip(
            "Quando ativo, envia {x, y} via UDP no canal de Rede e Eventos."
        )
        cl.addWidget(self.joy_udp_check)
        outer.addWidget(cfg_group)

        # === Visualização + Plot histórico ===
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        # Joystick widget
        self.joystick_widget = _VirtualJoystickWidget()
        split.addWidget(self.joystick_widget)
        # Plot X/Y temporal
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right); rl.setContentsMargins(2, 2, 2, 2)
        rl.addWidget(QtWidgets.QLabel("X/Y ao longo do tempo (últimos ~30s)"))
        self.joy_plot = pg.PlotWidget(enableMenu=False)
        self.joy_plot.showGrid(x=True, y=True, alpha=0.15)
        self.joy_plot.setLabel("left", "Eixo")
        self.joy_plot.setLabel("bottom", "Tempo", units="s")
        self.joy_plot.setMenuEnabled(False)
        self.joy_plot.addLegend(offset=(10, 10))
        self.joy_plot.setYRange(-1.1, 1.1)
        self.joy_x_curve = self.joy_plot.plot(
            pen=pg.mkPen(SIGNAL_TYPE_COLORS["EMG"], width=1.6), name="X")
        self.joy_y_curve = self.joy_plot.plot(
            pen=pg.mkPen("#66ddff", width=1.6), name="Y")
        rl.addWidget(self.joy_plot)
        split.addWidget(right)
        split.setSizes([300, 700])
        outer.addWidget(split, stretch=1)

        # Estado interno
        self._joy_history = {"t": [], "x": [], "y": []}
        self._joy_smooth = []  # buffer para smoothing
        return widget

    def _joy_repopulate_combos(self):
        """Popula combos de joystick com canais EMG disponíveis."""
        if not hasattr(self, "_joy_axes"): return
        emg_channels = [(ch, f"CH{ch+1} ({self.config.channel_mapping[ch]})")
                        for ch in range(MAX_CHANNELS)
                        if ch < len(self.config.channel_signal_types)
                        and self.config.channel_signal_types[ch] == "EMG"]
        # Defaults sugeridos
        defaults = {"+x": 0, "-x": 1, "+y": 2, "-y": 3}
        for code, ax in self._joy_axes.items():
            cb = ax["combo"]
            cb.blockSignals(True)
            cb.clear()
            if not emg_channels:
                cb.addItem("(nenhum canal EMG)", -1)
            else:
                for ch, label in emg_channels:
                    cb.addItem(label, ch)
                d = defaults.get(code, 0)
                if d < len(emg_channels):
                    cb.setCurrentIndex(d)
            cb.blockSignals(False)

    def _joy_calibrate(self, code):
        """Inicia calibração 3s do canal mapeado para uma direção."""
        if code not in self._joy_axes: return
        ax = self._joy_axes[code]
        ch = ax["combo"].currentData()
        if ch is None or ch < 0:
            QtWidgets.QMessageBox.warning(
                self, "Sem canal", "Selecione um canal EMG primeiro."); return
        ax["calibrating"] = True
        ax["calibration_data"] = []
        ax["calibration_start"] = time.time()
        self._log(f"Calibrando {code}: contraia músculo por 3s...")

    def _update_emg_joystick_view(self):
        """Atualiza joystick: lê envelope dos canais mapeados, normaliza, plota."""
        if not hasattr(self, "_joy_axes"): return
        if not hasattr(self, "emg_rows"):
            return  # depende da aba EMG ter sido construída
        # Lê envelope atual dos canais mapeados (do bar da aba EMG)
        method = self.config.emg_envelope_method
        window_ms = self.config.emg_envelope_window_ms
        data = self._ordered_buffer()
        if data.shape[1] < int(SAMPLE_RATE * 0.1): return
        env_by_ch = {}
        for code, ax in self._joy_axes.items():
            ch = ax["combo"].currentData()
            if ch is None or ch < 0: continue
            if ch in env_by_ch:
                env_val = env_by_ch[ch]
            else:
                env_signal = self._compute_emg_envelope(data[ch], SAMPLE_RATE, method, window_ms)
                env_val = float(env_signal[-1]) if len(env_signal) > 0 else 0.0
                env_by_ch[ch] = env_val
            ax["env_lbl"].setText(f"{env_val:.1f}")
            # Calibração: coleta máximo durante 3s
            if ax["calibrating"]:
                ax["calibration_data"].append(env_val)
                if (time.time() - ax["calibration_start"]) > 3.0:
                    if ax["calibration_data"]:
                        ax["max_value"] = float(np.percentile(ax["calibration_data"], 90))
                        ax["max_lbl"].setText(f"{ax['max_value']:.1f}")
                        self._log(f"{code} calibrado: max={ax['max_value']:.1f} µV")
                    ax["calibrating"] = False
                    ax["calibration_data"] = []
        # Calcula axes [-1, +1]
        def norm_val(code):
            ax = self._joy_axes[code]
            ch = ax["combo"].currentData()
            if ch is None or ch < 0: return 0.0
            env = env_by_ch.get(ch, 0.0)
            mx = max(ax["max_value"], 1e-6)
            return min(1.0, env / mx)
        pos_x = norm_val("+x"); neg_x = norm_val("-x")
        pos_y = norm_val("+y"); neg_y = norm_val("-y")
        x = pos_x - neg_x
        y = pos_y - neg_y
        # Dead zone
        dz = self.joy_dead_zone_spin.value()
        if abs(x) < dz: x = 0.0
        else: x = (x - dz * np.sign(x)) / (1.0 - dz)
        if abs(y) < dz: y = 0.0
        else: y = (y - dz * np.sign(y)) / (1.0 - dz)
        # Smoothing (moving average)
        sm_n = self.joy_smoothing_spin.value()
        self._joy_smooth.append((x, y))
        if len(self._joy_smooth) > sm_n:
            self._joy_smooth = self._joy_smooth[-sm_n:]
        if self._joy_smooth:
            arr = np.array(self._joy_smooth)
            x_s = float(np.mean(arr[:, 0]))
            y_s = float(np.mean(arr[:, 1]))
        else:
            x_s, y_s = x, y
        # Atualiza widget
        self.joystick_widget.set_axes(x_s, y_s, dz)
        # Histórico
        now = time.time()
        H = self._joy_history
        H["t"].append(now); H["x"].append(x_s); H["y"].append(y_s)
        if len(H["t"]) > 300:
            H["t"] = H["t"][-300:]
            H["x"] = H["x"][-300:]
            H["y"] = H["y"][-300:]
        t0 = H["t"][0]
        t_rel = [t - t0 for t in H["t"]]
        self.joy_x_curve.setData(t_rel, H["x"])
        self.joy_y_curve.setData(t_rel, H["y"])
        # UDP opcional
        if self.joy_udp_check.isChecked() and getattr(self, "udp", None) and self.udp.enabled:
            try:
                self.udp.send_sample(time.time(), [x_s, y_s])
            except Exception:
                pass

    # ---- Tab: Tempo Real ----
    def _build_realtime_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5); layout.setSpacing(4)

        # --- Barra de controle ---
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Visualização:"))
        self.rt_view_combo = QtWidgets.QComboBox()
        self.rt_view_combo.addItems([
            "Empilhado (todos visíveis)",
            "Individual (1 plot por canal, rolável)",
        ])
        self.rt_view_combo.currentIndexChanged.connect(self._on_rt_view_changed)
        ctrl.addWidget(self.rt_view_combo)

        ctrl.addSpacing(16)
        ctrl.addWidget(QtWidgets.QLabel("Escala:"))
        self.rt_scale_spin = QtWidgets.QDoubleSpinBox()
        self.rt_scale_spin.setRange(2.0, 5000.0)
        self.rt_scale_spin.setValue(100.0)
        self.rt_scale_spin.setDecimals(0)
        self.rt_scale_spin.setSingleStep(10.0)
        self.rt_scale_spin.setSuffix(" µV/canal")
        self.rt_scale_spin.setToolTip("Quantos µV ocupam o espaço de 1 canal. "
                                       "Menor = ondas maiores.")
        ctrl.addWidget(self.rt_scale_spin)
        for label, val in (("50", 50), ("100", 100), ("200", 200), ("500", 500)):
            b = QtWidgets.QPushButton(label)
            b.setMaximumWidth(42)
            b.setToolTip(f"Escala {val} µV/canal")
            b.clicked.connect(lambda _ck, v=val: self.rt_scale_spin.setValue(v))
            ctrl.addWidget(b)
        b_auto = QtWidgets.QPushButton("Auto")
        b_auto.setMaximumWidth(48)
        b_auto.setToolTip("Ajusta a escala automaticamente ao sinal atual")
        b_auto.clicked.connect(self._rt_autoscale)
        ctrl.addWidget(b_auto)

        ctrl.addSpacing(16)
        ctrl.addWidget(QtWidgets.QLabel("Janela:"))
        self.rt_window_spin = QtWidgets.QSpinBox()
        self.rt_window_spin.setRange(1, BUFFER_SECONDS)
        self.rt_window_spin.setValue(5)
        self.rt_window_spin.setSuffix(" s")
        ctrl.addWidget(self.rt_window_spin)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.setHandleWidth(4)

        # Stack: 0 = Empilhado (montagem clínica), 1 = Individual (rolável)
        self.rt_view_stack = QtWidgets.QStackedWidget()

        # ===== VIEW 0: EMPILHADO (single plot, offset vertical) =====
        self.montage_plot = pg.PlotWidget(enableMenu=False)
        self.montage_plot.showGrid(x=True, y=False, alpha=0.12)
        self.montage_plot.setMenuEnabled(False)
        self.montage_plot.setMouseEnabled(x=False, y=False)
        self.montage_plot.setLabel("bottom", "Tempo", units="s")
        self.montage_plot.getAxis("left").setWidth(70)
        self.montage_plot.hideButtons()
        self.montage_curves = []
        for ch in range(MAX_CHANNELS):
            cur = self.montage_plot.plot(
                pen=pg.mkPen(CHANNEL_COLORS[ch], width=1.1))
            self.montage_curves.append(cur)
        self.montage_marker_lines = []
        self.rt_view_stack.addWidget(self.montage_plot)

        # ===== VIEW 1: INDIVIDUAL (scrollable) =====
        self.channels_scroll = QtWidgets.QScrollArea()
        self.channels_scroll.setWidgetResizable(True)
        self.channels_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.channels_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.channels_layout = pg.GraphicsLayoutWidget()
        try:
            self.channels_layout.ci.layout.setVerticalSpacing(6)
        except Exception: pass
        self._channels_per_plot_height = 80
        self.channels_layout.setMinimumHeight(
            BASE_CHANNELS * self._channels_per_plot_height)
        self.channels_scroll.setWidget(self.channels_layout)

        self.channel_plots  = []
        self.channel_curves = []
        self.channel_marker_lines = [[] for _ in range(MAX_CHANNELS)]
        for ch in range(MAX_CHANNELS):
            plot = self.channels_layout.addPlot(row=ch, col=0, enableMenu=False)
            plot.showGrid(x=True, y=True, alpha=0.15)
            plot.setLabel("left", f"CH{ch + 1}", units="uV", color=CHANNEL_COLORS[ch])
            left_axis = plot.getAxis("left")
            left_axis.setTextPen(CHANNEL_COLORS[ch])
            left_axis.setWidth(64)
            plot.setMouseEnabled(x=False, y=True)
            plot.setMenuEnabled(False)
            plot.enableAutoRange(axis="y", enable=True)
            plot.setAutoVisible(y=True)
            plot.setMinimumHeight(self._channels_per_plot_height - 4)
            if ch > 0:
                plot.setXLink(self.channel_plots[0])
            plot.setLabel("bottom", "Tempo", units="s")
            plot.getAxis("bottom").setHeight(22)
            curve = plot.plot(pen=pg.mkPen(CHANNEL_COLORS[ch], width=1.2))
            curve.setDownsampling(auto=True, method="peak")  # menos CPU sem perda visual
            curve.setClipToView(True)
            self.channel_plots.append(plot)
            self.channel_curves.append(curve)
        self.rt_view_stack.addWidget(self.channels_scroll)

        self.rt_view_stack.setCurrentIndex(0)  # default: Empilhado
        splitter.addWidget(self.rt_view_stack)

        # Acelerômetro
        accel_widget = QtWidgets.QWidget()
        al = QtWidgets.QVBoxLayout(accel_widget)
        al.setContentsMargins(5, 5, 5, 5)
        accel_title = QtWidgets.QLabel("Acelerômetro (g — eixos X / Y / Z)")
        accel_title.setStyleSheet(f"color: {COLORS['text_dim']}; padding: 2px 8px;")
        al.addWidget(accel_title)
        self.accel_plot = pg.PlotWidget(enableMenu=False)
        self.accel_plot.showGrid(x=True, y=True, alpha=0.15)
        self.accel_plot.setLabel("left", "g")
        self.accel_plot.setLabel("bottom", "Tempo", units="s")
        self.accel_plot.setMenuEnabled(False)
        self.accel_plot.addLegend(offset=(10, 10))
        accel_colors = ["#ff5555", "#55ff55", "#5599ff"]
        self.accel_curves = []
        for i, axis in enumerate(("X", "Y", "Z")):
            curve = self.accel_plot.plot(pen=pg.mkPen(accel_colors[i], width=1.4), name=axis)
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)
            self.accel_curves.append(curve)
        al.addWidget(self.accel_plot)
        splitter.addWidget(accel_widget)
        splitter.setSizes([760, 180])
        layout.addWidget(splitter)
        return widget

    def _on_rt_view_changed(self, idx):
        self.rt_view_stack.setCurrentIndex(idx)
        # Limpa linhas de marker do modo montagem ao trocar
        if idx == 0 and hasattr(self, "montage_plot"):
            self._refresh_montage_layout()

    def _rt_autoscale(self):
        """Ajusta a escala (µV/canal) com base no desvio padrão atual."""
        data = self._ordered_buffer()
        if data.shape[1] < 10:
            return
        vis = [ch for ch in range(MAX_CHANNELS)
               if ch < self.num_channels and self.channel_active[ch]]
        if not vis:
            return
        # 4x o maior desvio padrão entre canais visíveis
        sd = max(float(np.std(data[ch])) for ch in vis)
        scale = max(10.0, min(5000.0, sd * 4.0))
        self.rt_scale_spin.setValue(round(scale))

    def _refresh_montage_layout(self):
        """Recalcula os ticks do eixo Y (nomes de canais) do modo Empilhado."""
        if not hasattr(self, "montage_plot"):
            return
        vis = [ch for ch in range(MAX_CHANNELS)
               if ch < self.num_channels and self.channel_active[ch]]
        n_vis = len(vis)
        ticks = []
        for k, ch in enumerate(vis):
            baseline = (n_vis - 1 - k)
            name = (self.config.channel_mapping[ch]
                    if ch < len(self.config.channel_mapping) else f"CH{ch+1}")
            ticks.append((baseline, f"CH{ch+1} {name}"))
        ax = self.montage_plot.getAxis("left")
        ax.setTicks([ticks])
        self.montage_plot.setYRange(-0.6, max(0.6, n_vis - 0.4), padding=0)

    # ---- Tab: Análises ----
    def _build_analysis_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(10)

        sel_row = QtWidgets.QHBoxLayout()
        sel_row.addWidget(QtWidgets.QLabel("Canal para FFT/Banda:"))
        self.analysis_channel = QtWidgets.QComboBox()
        for i in range(MAX_CHANNELS):
            self.analysis_channel.addItem(f"CH{i + 1}")
        self.analysis_channel.setMinimumWidth(120)
        sel_row.addWidget(self.analysis_channel); sel_row.addStretch()
        layout.addLayout(sel_row)

        graph_row = QtWidgets.QHBoxLayout()
        fft_group = QtWidgets.QGroupBox("Espectro de Frequência (FFT)")
        fft_layout = QtWidgets.QVBoxLayout(fft_group)
        self.fft_plot = pg.PlotWidget(enableMenu=False)
        self.fft_plot.showGrid(x=True, y=True, alpha=0.15)
        self.fft_plot.setLabel("left", "Amplitude", units="uV")
        self.fft_plot.setLabel("bottom", "Frequência", units="Hz")
        self.fft_plot.setXRange(0, 60); self.fft_plot.setMenuEnabled(False)
        self.fft_curve = self.fft_plot.plot(pen=pg.mkPen(COLORS["accent"], width=1.4))
        for _, (low, _h) in EEG_BANDS.items():
            line = pg.InfiniteLine(pos=low, angle=90,
                pen=pg.mkPen(COLORS["border"], style=QtCore.Qt.PenStyle.DashLine))
            self.fft_plot.addItem(line)
        fft_layout.addWidget(self.fft_plot)
        graph_row.addWidget(fft_group)

        band_group = QtWidgets.QGroupBox("Bandas de Potência EEG")
        band_layout = QtWidgets.QVBoxLayout(band_group)
        self.band_plot = pg.PlotWidget(enableMenu=False)
        self.band_plot.showGrid(x=False, y=True, alpha=0.15)
        self.band_plot.setLabel("left", "Potência", units="µV²/Hz")
        self.band_plot.setMenuEnabled(False)
        self.band_plot.getAxis("bottom").setTicks([list(enumerate(EEG_BANDS.keys()))])
        self.band_bars = pg.BarGraphItem(
            x=list(range(len(EEG_BANDS))),
            height=[0.0] * len(EEG_BANDS), width=0.6,
            brush=COLORS["accent"], pen=pg.mkPen(COLORS["accent_dim"]),
        )
        self.band_plot.addItem(self.band_bars)
        self.band_plot.setXRange(-0.5, len(EEG_BANDS) - 0.5)
        band_layout.addWidget(self.band_plot)
        graph_row.addWidget(band_group)
        layout.addLayout(graph_row, stretch=2)

        # Tabela de estatísticas — todos os canais (compacta, 16 linhas visíveis sem scroll)
        stats_group = QtWidgets.QGroupBox("Estatísticas — Todos os Canais")
        stats_layout = QtWidgets.QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(8, 4, 8, 4)
        self.stats_table = QtWidgets.QTableWidget(MAX_CHANNELS, 4)
        self.stats_table.setHorizontalHeaderLabels(
            ["Canal", "Média (µV)", "Desvio Padrão (µV)", "RMS (µV)"])
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        # Compacta: ~22px por linha -> 16 linhas + header cabem em ~380px
        STATS_ROW_H = 22
        self.stats_table.verticalHeader().setDefaultSectionSize(STATS_ROW_H)
        self.stats_table.horizontalHeader().setFixedHeight(26)
        self.stats_table.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.stats_table.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for ch in range(MAX_CHANNELS):
            name_item = QtWidgets.QTableWidgetItem(f"CH{ch + 1}")
            name_item.setForeground(QtGui.QColor(CHANNEL_COLORS[ch]))
            name_item.setFont(QtGui.QFont(FONT_DATA, 10, QtGui.QFont.Weight.Bold))
            self.stats_table.setItem(ch, 0, name_item)
            for col in range(1, 4):
                it = QtWidgets.QTableWidgetItem("--")
                it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.stats_table.setItem(ch, col, it)
        stats_layout.addWidget(self.stats_table)
        layout.addWidget(stats_group, stretch=1)
        return widget

    # ---- Tab: Topografia ----
    def _build_topology_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(10)

        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.addWidget(QtWidgets.QLabel("Banda no Head Plot:"))
        self.topo_band_combo = QtWidgets.QComboBox()
        self.topo_band_combo.addItems(list(EEG_BANDS.keys()))
        self.topo_band_combo.setCurrentText("Alpha")
        ctrl_row.addWidget(self.topo_band_combo)
        ctrl_row.addSpacing(16)
        ctrl_row.addWidget(QtWidgets.QLabel("Mapa:"))
        self.topo_mode_combo = QtWidgets.QComboBox()
        self.topo_mode_combo.addItems(["Interpolado (potencial)",
                                       "CSD / Laplaciano (fonte)"])
        self.topo_mode_combo.setToolTip(
            "Interpolado = potência de escalpo (IDW).\n"
            "CSD/Laplaciano = realça fontes locais e reduz condução de volume\n"
            "(McFarland 1997; Nunez & Srinivasan 2006).")
        self.topo_mode_combo.currentTextChanged.connect(
            lambda t: self.head_plot.set_map_mode("csd" if "CSD" in t else "interp"))
        ctrl_row.addWidget(self.topo_mode_combo)
        self.topo_loreta_btn = QtWidgets.QPushButton("vs LORETA / MNE…")
        self.topo_loreta_btn.setToolTip("Como este mapa se compara a LORETA/sLORETA "
                                        "e localização de fonte 3D.")
        self.topo_loreta_btn.clicked.connect(self._show_topo_methods_info)
        ctrl_row.addWidget(self.topo_loreta_btn)
        ctrl_row.addStretch()
        ctrl_row.addWidget(QtWidgets.QLabel("Canal para Focus:"))
        self.topo_channel_combo = QtWidgets.QComboBox()
        for i in range(MAX_CHANNELS):
            self.topo_channel_combo.addItem(f"CH{i+1}")
        ctrl_row.addWidget(self.topo_channel_combo)
        layout.addLayout(ctrl_row)

        main_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        head_group = QtWidgets.QGroupBox("Head Plot — Potência por Eletrodo (10-20)")
        hl = QtWidgets.QVBoxLayout(head_group)
        self.head_plot = HeadPlotWidget()
        hl.addWidget(self.head_plot)
        main_split.addWidget(head_group)

        side = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        focus_group = QtWidgets.QGroupBox("Focus")
        fl = QtWidgets.QVBoxLayout(focus_group)
        self.focus_meter = FocusMeterWidget()
        fl.addWidget(self.focus_meter)
        side.addWidget(focus_group)
        emg_group = QtWidgets.QGroupBox("EMG")
        el = QtWidgets.QVBoxLayout(emg_group)
        self.emg_widget = EMGEnvelopeWidget()
        el.addWidget(self.emg_widget)
        side.addWidget(emg_group)
        side.setSizes([300, 400])
        main_split.addWidget(side)
        main_split.setSizes([800, 600])
        layout.addWidget(main_split)
        return widget

    def _show_topo_methods_info(self):
        """Comparação honesta: topomap × CSD × LORETA/sLORETA × MNE/EEGLAB."""
        acc = COLORS.get("accent", "#0f9d75")
        html = (
            "<h3 style='color:%s'>Análise visual: o que cada método mostra</h3>"
            "<p><b>1) Interpolado (potencial):</b> projeta a potência por eletrodo "
            "numa vista da cabeça (IDW). Rápido para ver <i>onde</i> há atividade, "
            "mas é <b>potencial de escalpo</b> — sofre de <b>condução de volume</b> "
            "(a fonte real fica 'borrada').</p>"
            "<p><b>2) CSD / Laplaciano de superfície</b> (este software): realça "
            "<b>fontes locais</b> e reduz a condução de volume usando só a geometria "
            "dos eletrodos — é uma referência espacial 'sem referência'. Melhora muito "
            "a nitidez do mapa. <i>[McFarland 1997; Nunez &amp; Srinivasan 2006]</i></p>"
            "<p><b>3) LORETA / sLORETA</b> (localização de fonte 3D): estima a "
            "distribuição de corrente <b>dentro do volume cerebral</b>. LORETA assume "
            "solução suave; sLORETA é padronizada, com erro de localização zero para "
            "fonte única. Requer <b>modelo de cabeça/leadfield</b>. "
            "<i>[Pascual-Marqui 1994; 2002]</i></p>"
            "<hr><p><b>Comparação honesta:</b> o OpenBionica é um "
            "coletor+analisador leve (topomap 2D, CSD, FFT, bandas, ERS/ERD). "
            "Tomografia de fonte 3D (LORETA/sLORETA) é o domínio do "
            "<b>MNE-Python</b> <i>[Gramfort 2013]</i> e do <b>EEGLAB</b> "
            "<i>[Delorme 2004]</i>.</p>"
            "<p><b>Caminho recomendado:</b> use o OpenBionica para "
            "coleta/triagem visual (incl. CSD) e <b>exporte para MNE/EEGLAB</b> "
            "(o software já exporta EDF/BIDS) quando precisar de fonte 3D.</p>"
            % acc)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Métodos de análise visual — comparação")
        dlg.setMinimumSize(600, 520)
        v = QtWidgets.QVBoxLayout(dlg)
        tb = QtWidgets.QTextBrowser(); tb.setHtml(html); tb.setOpenExternalLinks(True)
        v.addWidget(tb, 1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(dlg.reject); bb.accepted.connect(dlg.accept)
        v.addWidget(bb)
        try: dlg.setStyleSheet(build_stylesheet(COLORS))
        except Exception: pass
        dlg.exec()

    # ---- Tab: Espectrograma ----
    def _build_spectrogram_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        sel_row = QtWidgets.QHBoxLayout()
        sel_row.addWidget(QtWidgets.QLabel("Canal:"))
        self.spec_channel = QtWidgets.QComboBox()
        for i in range(MAX_CHANNELS):
            self.spec_channel.addItem(f"CH{i + 1}")
        self.spec_channel.setMinimumWidth(120)
        sel_row.addWidget(self.spec_channel); sel_row.addStretch()
        sel_row.addWidget(QtWidgets.QLabel(
            f"Janela: {SPEC_FRAMES * 0.25:.0f}s   Freq: 0–{SPEC_FMAX} Hz   dB normalizado"))
        layout.addLayout(sel_row)
        self.spec_widget = pg.PlotWidget(enableMenu=False)
        self.spec_widget.setLabel("left", "Frequência", units="Hz")
        self.spec_widget.setLabel("bottom", "Tempo", units="s")
        self.spec_widget.setMenuEnabled(False)
        self.spec_image = pg.ImageItem()
        stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        colors_m = np.array([
            [0,   0,   0,   255],
            [80,  10,  80,  255],
            [160, 30,  60,  255],
            [255, 130, 30,  255],
            [255, 255, 180, 255],
        ], dtype=np.ubyte)
        cmap = pg.ColorMap(stops, colors_m)
        self.spec_image.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        self.spec_image.setLevels([-80, 0])
        self.spec_widget.addItem(self.spec_image)
        self.spec_widget.setYRange(0, SPEC_FMAX)
        self.spec_widget.setXRange(0, SPEC_FRAMES * 0.25)
        layout.addWidget(self.spec_widget)
        return widget

    # ---- Tab: Filtros & Canais ----
    def _build_filters_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(15, 15, 15, 15); layout.setSpacing(12)

        # ============================================================
        # Re-referenciação (CAR / Laplacian / Mastoide / Bipolar / REST)
        # ============================================================
        reref_group = QtWidgets.QGroupBox(
            "Re-referenciação (aplicada ANTES dos filtros)"
        )
        reref_l = QtWidgets.QHBoxLayout(reref_group)
        reref_l.addWidget(QtWidgets.QLabel("Esquema:"))
        self.reref_combo = QtWidgets.QComboBox()
        # (label visível, código interno)
        self.reref_combo.addItem("Nenhuma (sinal original)",            "none")
        self.reref_combo.addItem("CAR — Common Average Reference",      "car")
        self.reref_combo.addItem("Mastoide — média de canais ref. (M1+M2)/2",
                                  "mastoid")
        self.reref_combo.addItem("Laplaciano de Superfície (4 vizinhos)",
                                  "laplacian")
        self.reref_combo.addItem("Bipolar (ch_i − ch_i+1)",             "bipolar")
        self.reref_combo.addItem("REST (canais distantes — aprox.)",    "rest")
        self.reref_combo.setToolTip(
            "Esquema de re-referência aplicado em tempo real:\n"
            "  CAR        : média de TODOS os canais subtraída (padrão para 32+ ch)\n"
            "  Mastoide   : usa eletrodos M1/M2 como referência clínica\n"
            "  Laplaciano : sinal de cada canal menos média dos 4 vizinhos\n"
            "  Bipolar    : útil para EMG (diferença entre eletrodos do par)\n"
            "  REST       : aproximação ZERO reference (Yao 2001)"
        )
        self.reref_combo.currentIndexChanged.connect(self._on_reref_changed)
        reref_l.addWidget(self.reref_combo, stretch=1)
        # Campo de canais de referência (para Mastoid)
        reref_l.addWidget(QtWidgets.QLabel("Canais ref. (1-based, vírgula):"))
        self.reref_channels_edit = QtWidgets.QLineEdit()
        self.reref_channels_edit.setPlaceholderText("Ex.: 13,14 (TP9, TP10)")
        self.reref_channels_edit.setMaximumWidth(160)
        self.reref_channels_edit.editingFinished.connect(self._on_reref_changed)
        reref_l.addWidget(self.reref_channels_edit)
        layout.addWidget(reref_group)

        notch_group = QtWidgets.QGroupBox("Filtro Notch (rejeição de banda)")
        nl = QtWidgets.QHBoxLayout(notch_group)
        self.notch_enable = QtWidgets.QCheckBox("Ativado")
        self.notch_enable.toggled.connect(self._on_filter_change)
        nl.addWidget(self.notch_enable); nl.addSpacing(20)
        nl.addWidget(QtWidgets.QLabel("Frequência (Hz):"))
        self.notch_freq = QtWidgets.QComboBox()
        self.notch_freq.addItems(["50", "60"]); self.notch_freq.setCurrentText("60")
        self.notch_freq.currentTextChanged.connect(self._on_filter_change)
        nl.addWidget(self.notch_freq); nl.addStretch()
        nl.addWidget(QtWidgets.QLabel("(Q=30 — remove ruido de rede elétrica)"))
        layout.addWidget(notch_group)

        bp_group = QtWidgets.QGroupBox("Filtro Bandpass (Butterworth, ordem 4)")
        bp_outer = QtWidgets.QVBoxLayout(bp_group)
        bl = QtWidgets.QHBoxLayout()
        self.bp_enable = QtWidgets.QCheckBox("Ativado")
        self.bp_enable.toggled.connect(self._on_filter_change)
        bl.addWidget(self.bp_enable); bl.addSpacing(20)
        bl.addWidget(QtWidgets.QLabel("Corte inferior (Hz):"))
        self.bp_low = QtWidgets.QDoubleSpinBox()
        self.bp_low.setRange(0.1, 100.0); self.bp_low.setDecimals(1); self.bp_low.setSingleStep(0.5)
        self.bp_low.setValue(1.0)
        self.bp_low.valueChanged.connect(self._on_filter_change)
        bl.addWidget(self.bp_low); bl.addSpacing(15)
        bl.addWidget(QtWidgets.QLabel("Corte superior (Hz):"))
        self.bp_high = QtWidgets.QDoubleSpinBox()
        self.bp_high.setRange(0.5, 120.0); self.bp_high.setDecimals(1); self.bp_high.setSingleStep(0.5)
        self.bp_high.setValue(50.0)
        self.bp_high.valueChanged.connect(self._on_filter_change)
        bl.addWidget(self.bp_high); bl.addStretch()
        bp_outer.addLayout(bl)
        presets = QtWidgets.QHBoxLayout()
        presets.addWidget(QtWidgets.QLabel("Presets:"))
        for name, low, high in (("Largo 1-50", 1, 50), ("Mu 8-13", 8, 13),
                                 ("Beta 13-30", 13, 30), ("Gamma 30-50", 30, 50)):
            btn = QtWidgets.QPushButton(name)
            btn.clicked.connect(lambda _ck, l=low, h=high: self._apply_bp_preset(l, h))
            presets.addWidget(btn)
        presets.addStretch()
        bp_outer.addLayout(presets)
        layout.addWidget(bp_group)

        # Configuração por canal: ativo + tipo de sinal (multimodal)
        ch_group = QtWidgets.QGroupBox(
            "Canais — Ativação e Tipo de Sinal (multimodal: EEG / EMG / ECG / EoG)"
        )
        cl_outer = QtWidgets.QVBoxLayout(ch_group)
        # Botões rápidos para presets
        preset_row = QtWidgets.QHBoxLayout()
        preset_row.addWidget(QtWidgets.QLabel("Presets:"))
        for label, setter in (
            ("Todos EEG",  lambda: self._apply_signal_type_to_all("EEG")),
            ("Todos EMG",  lambda: self._apply_signal_type_to_all("EMG")),
            ("Todos ECG",  lambda: self._apply_signal_type_to_all("ECG")),
            ("Todos EoG",  lambda: self._apply_signal_type_to_all("EoG")),
            ("1-8 EEG / 9-16 EMG", lambda: self._apply_signal_type_split(8, "EEG", "EMG")),
        ):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(setter)
            preset_row.addWidget(b)
        preset_row.addStretch()
        cl_outer.addLayout(preset_row)

        cl = QtWidgets.QGridLayout()
        cl.setHorizontalSpacing(6); cl.setVerticalSpacing(2)
        # Cabeçalhos: 2 blocos de 8 canais lado a lado (CH1-8 | CH9-16) para evitar rolagem
        for block in range(2):
            base_col = block * 5
            hdrs = ["Ativo", "Canal", "Tipo", "Filtro recomendado", "Threshold EMG"]
            for i, h in enumerate(hdrs):
                lbl = QtWidgets.QLabel(h)
                lbl.setStyleSheet(f"color: {COLORS['accent']}; font-weight: bold;")
                cl.addWidget(lbl, 0, base_col + i)

        self.channel_checks = []
        self.channel_type_combos = []
        self.channel_filter_hint_lbls = []
        self.channel_emg_thresh_inline = []
        for ch in range(MAX_CHANNELS):
            block = ch // 8  # 0 → coluna esquerda; 1 → coluna direita
            row = (ch % 8) + 1
            base_col = block * 5
            # Ativo
            cb = QtWidgets.QCheckBox()
            cb.setChecked(True)
            cb.setStyleSheet(f"QCheckBox {{ color: {CHANNEL_COLORS[ch]}; }}")
            cb.toggled.connect(lambda checked, c=ch: self._on_channel_toggle(c, checked))
            cl.addWidget(cb, row, base_col + 0)
            self.channel_checks.append(cb)
            # Nome canal
            ch_lbl = QtWidgets.QLabel(f"CH{ch+1}")
            ch_lbl.setStyleSheet(f"color: {CHANNEL_COLORS[ch]}; font-weight: bold;")
            ch_lbl.setMinimumWidth(46)
            cl.addWidget(ch_lbl, row, base_col + 1)
            # Tipo (combo)
            tc = QtWidgets.QComboBox()
            for stype in SIGNAL_TYPES:
                tc.addItem(stype)
            cur_type = (self.config.channel_signal_types[ch]
                        if ch < len(self.config.channel_signal_types) else "EEG")
            tc.setCurrentText(cur_type)
            tc.currentTextChanged.connect(
                lambda t, c=ch: self._on_channel_signal_type_changed(c, t))
            cl.addWidget(tc, row, base_col + 2)
            self.channel_type_combos.append(tc)
            # Filtro recomendado (dinâmico)
            preset = SIGNAL_FILTER_PRESETS.get(cur_type, SIGNAL_FILTER_PRESETS["EEG"])
            hint = QtWidgets.QLabel(preset["label"])
            hint.setStyleSheet(
                f"color: {SIGNAL_TYPE_COLORS.get(cur_type, COLORS['text_dim'])}; "
                f"font-size: 9pt;")
            hint.setMinimumWidth(140)
            cl.addWidget(hint, row, base_col + 3)
            self.channel_filter_hint_lbls.append(hint)
            # Threshold EMG inline (visível se canal é EMG)
            th_inline = QtWidgets.QDoubleSpinBox()
            th_inline.setRange(1.0, 5000.0)
            th_inline.setDecimals(1)
            th_inline.setSingleStep(5.0)
            th_inline.setValue(self.config.emg_threshold_uV[ch]
                               if ch < len(self.config.emg_threshold_uV) else 50.0)
            th_inline.setSuffix(" µV")
            th_inline.setEnabled(cur_type == "EMG")
            th_inline.valueChanged.connect(
                lambda v, c=ch: self._emg_threshold_changed_inline(c, v))
            cl.addWidget(th_inline, row, base_col + 4)
            self.channel_emg_thresh_inline.append(th_inline)

        cl_outer.addLayout(cl)

        # Botão: aplicar preset de filtro do canal selecionado
        f_row = QtWidgets.QHBoxLayout()
        apply_preset_btn = QtWidgets.QPushButton("Aplicar filtro do tipo de sinal (global)")
        apply_preset_btn.setToolTip(
            "Aplica os parâmetros bandpass + notch recomendados para o tipo de sinal "
            "predominante. Se houver mistura EEG/EMG, usa o tipo da maioria."
        )
        apply_preset_btn.clicked.connect(self._apply_dominant_signal_filter)
        f_row.addWidget(apply_preset_btn)
        f_row.addStretch()
        cl_outer.addLayout(f_row)

        layout.addWidget(ch_group)

        # ---- Detecção automática de canais ruins ----
        bad_group = QtWidgets.QGroupBox(
            "Detecção Automática de Canais Ruins (variância + correlação)"
        )
        bgl = QtWidgets.QHBoxLayout(bad_group)
        bgl.addWidget(QtWidgets.QLabel(
            "Identifica canais com variância anômala ou descorrelacionados "
            "dos vizinhos (eletrodo solto / saturação)."
        ))
        bgl.addStretch()
        self.bad_detect_btn = QtWidgets.QPushButton("Detectar agora")
        self.bad_detect_btn.clicked.connect(self._detect_bad_channels)
        bgl.addWidget(self.bad_detect_btn)
        self.bad_detect_status = QtWidgets.QLabel("--")
        self.bad_detect_status.setStyleSheet(f"color: {COLORS['text_dim']};")
        bgl.addWidget(self.bad_detect_status)
        layout.addWidget(bad_group)

        # ---- Modo de Aquisição (controla visibilidade de abas) ----
        mode_group = QtWidgets.QGroupBox(tr("Modo de Aquisição — Visibilidade de Abas"))
        mode_l = QtWidgets.QHBoxLayout(mode_group)
        mode_l.addWidget(QtWidgets.QLabel(tr(
            "Esconde abas que não fazem sentido para o modo escolhido "
            "(ex.: Topografia/ERP são EEG-only)."
        )))
        mode_l.addStretch()
        self.mode_visibility_combo = QtWidgets.QComboBox()
        for code, label in (("EEG", tr("Apenas EEG")),
                             ("EMG", tr("Apenas EMG")),
                             ("ECG", tr("Apenas ECG")),
                             ("EoG", tr("Apenas EoG")),
                             ("Hibrido", tr("Híbrido (multimodal)"))):
            self.mode_visibility_combo.addItem(label, code)
        # Default conforme tipo predominante
        try:
            from collections import Counter as _C
            types = [t for t in self.config.channel_signal_types if t != "off"]
            dom = _C(types).most_common(1)[0][0] if types else "EEG"
        except Exception:
            dom = "EEG"
        idx_d = self.mode_visibility_combo.findData(dom)
        if idx_d >= 0:
            self.mode_visibility_combo.setCurrentIndex(idx_d)
        mode_l.addWidget(self.mode_visibility_combo)
        apply_mode_btn = QtWidgets.QPushButton(tr("Aplicar modo"))
        apply_mode_btn.clicked.connect(self._on_mode_visibility_apply)
        mode_l.addWidget(apply_mode_btn)
        layout.addWidget(mode_group)

        layout.addStretch()
        return widget

    def _on_mode_visibility_apply(self):
        """Aplica o modo escolhido no combo da aba Filtros e Canais."""
        if not hasattr(self, "mode_visibility_combo"): return
        code = self.mode_visibility_combo.currentData()
        if not code: return
        self._apply_signal_mode_visibility(code)

    # ---- Handlers de tipo de sinal por canal ----
    def _on_channel_signal_type_changed(self, ch, new_type):
        """Salva o tipo de sinal escolhido para um canal."""
        if ch < 0 or ch >= MAX_CHANNELS: return
        if new_type not in SIGNAL_TYPES: return
        while len(self.config.channel_signal_types) <= ch:
            self.config.channel_signal_types.append("EEG")
        self.config.channel_signal_types[ch] = new_type
        self.config.save()
        # Atualiza hint de filtro nesta aba
        if hasattr(self, "channel_filter_hint_lbls"):
            preset = SIGNAL_FILTER_PRESETS.get(new_type, SIGNAL_FILTER_PRESETS["EEG"])
            self.channel_filter_hint_lbls[ch].setText(preset["label"])
            self.channel_filter_hint_lbls[ch].setStyleSheet(
                f"color: {SIGNAL_TYPE_COLORS.get(new_type, COLORS['text_dim'])}; font-size: 9pt;")
        if hasattr(self, "channel_emg_thresh_inline"):
            self.channel_emg_thresh_inline[ch].setEnabled(new_type == "EMG")
        # Reflete na aba EMG, se já construída
        if hasattr(self, "emg_rows"):
            self._emg_refresh_channel_types()
        # ECG/EoG combos dependem do tipo
        if hasattr(self, "ecg_channel_combo"):
            self._populate_ecg_channel_combo()
        if hasattr(self, "eog_h_combo"):
            self._populate_eog_channel_combos()
        # EMG Joystick combos
        if hasattr(self, "_joy_axes"):
            self._joy_repopulate_combos()
        self._log(f"CH{ch+1} agora é tipo {new_type}")
        self._audit_event("channel_type_change", channel=ch+1, type=new_type)

    def _apply_signal_type_to_all(self, stype):
        """Atribui o mesmo tipo a todos os canais."""
        if stype not in SIGNAL_TYPES: return
        for ch in range(MAX_CHANNELS):
            if hasattr(self, "channel_type_combos"):
                self.channel_type_combos[ch].setCurrentText(stype)
            else:
                self._on_channel_signal_type_changed(ch, stype)

    def _apply_signal_type_split(self, split_at, type_a, type_b):
        """Aplica type_a aos primeiros split_at canais e type_b aos restantes."""
        for ch in range(MAX_CHANNELS):
            stype = type_a if ch < split_at else type_b
            if hasattr(self, "channel_type_combos"):
                self.channel_type_combos[ch].setCurrentText(stype)
            else:
                self._on_channel_signal_type_changed(ch, stype)

    def _apply_dominant_signal_filter(self):
        """Aplica o preset de filtro do tipo majoritário entre os canais."""
        from collections import Counter
        types = [t for t in self.config.channel_signal_types if t != "off"]
        if not types:
            return
        dom = Counter(types).most_common(1)[0][0]
        preset = SIGNAL_FILTER_PRESETS.get(dom)
        if not preset: return
        # Aplica nos spinboxes de filtro
        if hasattr(self, "bp_low"):  self.bp_low.setValue(preset["hp"])
        if hasattr(self, "bp_high"): self.bp_high.setValue(preset["lp"])
        if hasattr(self, "bp_enable"): self.bp_enable.setChecked(True)
        if hasattr(self, "notch_enable"): self.notch_enable.setChecked(preset["notch"])
        self._on_filter_change()
        self._log(f"Filtro ajustado para {dom}: {preset['label']}")

    def _emg_threshold_changed_inline(self, ch, val):
        """Threshold ajustado na aba Filtros e Canais — sincroniza com aba EMG."""
        if ch < 0 or ch >= MAX_CHANNELS: return
        while len(self.config.emg_threshold_uV) <= ch:
            self.config.emg_threshold_uV.append(50.0)
        self.config.emg_threshold_uV[ch] = float(val)
        # Reflete na aba EMG (se construída)
        if hasattr(self, "emg_rows") and ch < len(self.emg_rows):
            spin = self.emg_rows[ch]["th_spin"]
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        if hasattr(self, "emg_threshold_lines"):
            self.emg_threshold_lines[ch].setPos(val)
        self.config.save()

    def _apply_bp_preset(self, low, high):
        self.bp_low.setValue(low); self.bp_high.setValue(high)
        self.bp_enable.setChecked(True); self._on_filter_change()

    # ---- Tab: Hardware ----
    def _build_hardware_tab(self):
        """Aba Hardware — mostra os 16 canais + controles sem rolagem."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(10, 8, 10, 8); layout.setSpacing(6)

        ch_group = QtWidgets.QGroupBox(
            "Configuração por Canal "
            "(formato comando: x CH PD GAIN INPUT BIAS SRB2 SRB1 X)"
        )
        cl = QtWidgets.QGridLayout(ch_group)
        cl.setHorizontalSpacing(6); cl.setVerticalSpacing(2)
        cl.setContentsMargins(8, 6, 8, 6)
        headers = ["Canal", "Power", "Gain", "Input", "Bias", "SRB2", "SRB1"]
        for i, h in enumerate(headers):
            lbl = QtWidgets.QLabel(h)
            lbl.setStyleSheet(f"color: {COLORS['accent']}; font-weight: bold;")
            cl.addWidget(lbl, 0, i)

        self.hw_power = []; self.hw_gain  = []; self.hw_input = []
        self.hw_bias  = []; self.hw_srb2  = []; self.hw_srb1  = []
        self.hw_row_widgets = []

        gain_opts = [("1x", "0"), ("2x", "1"), ("4x", "2"), ("6x", "3"),
                     ("8x", "4"), ("12x", "5"), ("24x", "6")]
        input_opts = [
            ("Normal",   "0"), ("Shorted",  "1"), ("BIAS_MEAS","2"),
            ("MVDD",     "3"), ("Temp",     "4"), ("Test",     "5"),
            ("BIAS_DRP", "6"), ("BIAS_DRN", "7"),
        ]
        power_opts = [("ON", "0"), ("OFF", "1")]
        binary_opts = [("Sim", "1"), ("Não", "0")]

        def make_combo(items, default_code):
            cb = QtWidgets.QComboBox()
            for label, code in items: cb.addItem(label, code)
            idx = next((i for i, (_, c) in enumerate(items) if c == default_code), 0)
            cb.setCurrentIndex(idx)
            cb.setMaximumHeight(24)  # compacto
            return cb

        for ch in range(MAX_CHANNELS):
            name = QtWidgets.QLabel(f"CH{ch + 1}")
            name.setStyleSheet(f"color: {CHANNEL_COLORS[ch]}; font-weight: bold;")
            row_widgets = [name]
            cl.addWidget(name, ch + 1, 0)
            pwr = make_combo(power_opts, "0"); cl.addWidget(pwr, ch + 1, 1); self.hw_power.append(pwr); row_widgets.append(pwr)
            g   = make_combo(gain_opts, "6");  cl.addWidget(g,   ch + 1, 2); self.hw_gain.append(g);   row_widgets.append(g)
            inp = make_combo(input_opts, "0"); cl.addWidget(inp, ch + 1, 3); self.hw_input.append(inp); row_widgets.append(inp)
            b   = make_combo(binary_opts, "1"); cl.addWidget(b,   ch + 1, 4); self.hw_bias.append(b);   row_widgets.append(b)
            s2  = make_combo(binary_opts, "1"); cl.addWidget(s2,  ch + 1, 5); self.hw_srb2.append(s2);  row_widgets.append(s2)
            s1  = make_combo(binary_opts, "0"); cl.addWidget(s1,  ch + 1, 6); self.hw_srb1.append(s1);  row_widgets.append(s1)
            self.hw_row_widgets.append(row_widgets)

        layout.addWidget(ch_group)

        # Botões + Quick + Manual em UMA linha cada (compactos)
        btn_row = QtWidgets.QHBoxLayout()
        apply_all = QtWidgets.QPushButton("Aplicar aos canais ATIVOS")
        apply_all.clicked.connect(self._apply_hardware_settings_all)
        btn_row.addWidget(apply_all)
        default_btn = QtWidgets.QPushButton("Restaurar padrão")
        default_btn.setToolTip("24x, normal, bias on, SRB2 on")
        default_btn.clicked.connect(self._restore_hw_defaults)
        btn_row.addWidget(default_btn)
        btn_row.addStretch()
        # Comandos rapidos na mesma linha
        for label, cmd, tip in (
            ("v", "v", "Versão do firmware"),
            ("?", "?", "Status dos canais"),
            ("d", "d", "Restaura defaults na placa"),
            ("[", "[", "Liga sinal de teste"),
            ("]", "]", "Desliga sinal de teste"),
            ("C", "C", "Expansão ON (16ch)"),
            ("c", "c", "Expansão OFF (8ch)"),
        ):
            b = QtWidgets.QPushButton(label)
            b.setMaximumWidth(36)
            b.setToolTip(f"Comando rápido: {tip}")
            b.clicked.connect(lambda _ck, c=cmd: self._send_quick_command(c))
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        # Comando manual em linha unica
        manual_row = QtWidgets.QHBoxLayout()
        manual_row.addWidget(QtWidgets.QLabel("Comando manual:"))
        self.hw_cmd_edit = QtWidgets.QLineEdit()
        self.hw_cmd_edit.setPlaceholderText("ex: x1060110X (CH1 normal 24x bias SRB2)")
        manual_row.addWidget(self.hw_cmd_edit, stretch=1)
        send_btn = QtWidgets.QPushButton("Enviar")
        send_btn.clicked.connect(self._send_hardware_command)
        manual_row.addWidget(send_btn)
        layout.addLayout(manual_row)
        return widget

    def _restore_hw_defaults(self):
        for ch in range(MAX_CHANNELS):
            self.hw_power[ch].setCurrentIndex(0)
            self.hw_gain[ch].setCurrentIndex(6)
            self.hw_input[ch].setCurrentIndex(0)
            self.hw_bias[ch].setCurrentIndex(0)
            self.hw_srb2[ch].setCurrentIndex(0)
            self.hw_srb1[ch].setCurrentIndex(1)

    # ---- Tab: Rede & Eventos ----
    def _build_network_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(15, 15, 15, 15); layout.setSpacing(12)

        udp_group = QtWidgets.QGroupBox("Streaming UDP (JSON)")
        ul = QtWidgets.QGridLayout(udp_group)
        ul.addWidget(QtWidgets.QLabel("Host:"), 0, 0)
        self.udp_host_edit = QtWidgets.QLineEdit("127.0.0.1")
        ul.addWidget(self.udp_host_edit, 0, 1)
        ul.addWidget(QtWidgets.QLabel("Porta:"), 0, 2)
        self.udp_port_spin = QtWidgets.QSpinBox()
        self.udp_port_spin.setRange(1, 65535); self.udp_port_spin.setValue(12345)
        ul.addWidget(self.udp_port_spin, 0, 3)
        self.udp_toggle_btn = QtWidgets.QPushButton("Iniciar streaming")
        self.udp_toggle_btn.setCheckable(True)
        self.udp_toggle_btn.clicked.connect(self._toggle_udp)
        ul.addWidget(self.udp_toggle_btn, 0, 4)
        info = QtWidgets.QLabel(
            "Envia cada amostra como UDP JSON {'t': float, 'n': int, 'v': [N floats]}. "
            "N varia conforme expansão (8 ou 16). "
            "Markers viram {'t': float, 'marker': str}.")
        info.setStyleSheet(f"color: {COLORS['text_dim']};"); info.setWordWrap(True)
        ul.addWidget(info, 1, 0, 1, 5)
        layout.addWidget(udp_group)

        # ===== LSL (Lab Streaming Layer) — padrão clínico/cientifico =====
        lsl_group = QtWidgets.QGroupBox("Streaming LSL (Lab Streaming Layer)")
        ll = QtWidgets.QGridLayout(lsl_group)
        ll.addWidget(QtWidgets.QLabel("Nome do stream:"), 0, 0)
        self.lsl_name_edit = QtWidgets.QLineEdit("EEG_Data_Collector")
        ll.addWidget(self.lsl_name_edit, 0, 1, 1, 2)
        self.lsl_toggle_btn = QtWidgets.QPushButton("Iniciar LSL")
        self.lsl_toggle_btn.setCheckable(True)
        self.lsl_toggle_btn.clicked.connect(self._toggle_lsl)
        if not HAS_LSL:
            self.lsl_toggle_btn.setEnabled(False)
            self.lsl_toggle_btn.setText("LSL indisponível (pip install pylsl)")
        ll.addWidget(self.lsl_toggle_btn, 0, 3)
        lsl_info = QtWidgets.QLabel(
            "Sincronização sub-milissegundo cross-aplicação. Padrão de fato em "
            "laboratórios de neurociência — funciona com PsychoPy, OpenViBE, "
            "BCI2000, MATLAB, Unity, etc. Cria 2 outlets: dados EEG + markers."
        )
        lsl_info.setStyleSheet(f"color: {COLORS['text_dim']};"); lsl_info.setWordWrap(True)
        ll.addWidget(lsl_info, 1, 0, 1, 4)
        layout.addWidget(lsl_group)

        # ===== LSL Receiver — recebe streams de outros apps =====
        lslr_group = QtWidgets.QGroupBox(
            "Receber via LSL (markers/eventos de PsychoPy, OpenViBE, etc.)"
        )
        lslr_l = QtWidgets.QVBoxLayout(lslr_group)
        lslr_row = QtWidgets.QHBoxLayout()
        self.lslr_resolve_btn = QtWidgets.QPushButton("Procurar streams LSL")
        self.lslr_resolve_btn.clicked.connect(self._lslr_resolve)
        if not HAS_LSL:
            self.lslr_resolve_btn.setEnabled(False)
            self.lslr_resolve_btn.setText("LSL indisponível (pip install pylsl)")
        lslr_row.addWidget(self.lslr_resolve_btn)
        lslr_row.addWidget(QtWidgets.QLabel("Stream:"))
        self.lslr_streams_combo = QtWidgets.QComboBox()
        self.lslr_streams_combo.setMinimumWidth(280)
        lslr_row.addWidget(self.lslr_streams_combo, stretch=1)
        self.lslr_subscribe_btn = QtWidgets.QPushButton("Assinar (markers)")
        self.lslr_subscribe_btn.setCheckable(True)
        self.lslr_subscribe_btn.clicked.connect(self._lslr_toggle_subscribe)
        if not HAS_LSL:
            self.lslr_subscribe_btn.setEnabled(False)
        lslr_row.addWidget(self.lslr_subscribe_btn)
        lslr_l.addLayout(lslr_row)
        self.lslr_status_lbl = QtWidgets.QLabel(
            "Pronto. Procure streams LSL na rede (ex.: PsychoPy emite markers do experimento)."
        )
        self.lslr_status_lbl.setStyleSheet(f"color: {COLORS['text_dim']};")
        self.lslr_status_lbl.setWordWrap(True)
        lslr_l.addWidget(self.lslr_status_lbl)
        layout.addWidget(lslr_group)

        # Estado interno do receiver
        self._lslr_streams_found = []   # lista de StreamInfo
        self._lslr_inlet = None         # pylsl.StreamInlet ativo
        self._lslr_timer = None         # QTimer que faz pull periódico

        mk_group = QtWidgets.QGroupBox("Marcadores / Eventos (hotkey global: M)")
        ml = QtWidgets.QVBoxLayout(mk_group)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Rótulo:"))
        self.marker_edit = QtWidgets.QLineEdit()
        self.marker_edit.setPlaceholderText("ex: olhos_abertos, estimulo_A, prova_1")
        row.addWidget(self.marker_edit, stretch=1)
        inj_btn = QtWidgets.QPushButton("Injetar marker")
        inj_btn.clicked.connect(self._inject_marker_from_edit)
        row.addWidget(inj_btn)
        ml.addLayout(row)

        quick = QtWidgets.QHBoxLayout()
        quick.addWidget(QtWidgets.QLabel("Atalhos rápidos:"))
        for label in ("REPOUSO", "ESTIMULO", "MOTOR", "PIQUE"):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(lambda _ck, lb=label: self._inject_marker_text(lb))
            quick.addWidget(b)
        quick.addStretch()
        ml.addLayout(quick)
        ml.addWidget(QtWidgets.QLabel("Marcadores recentes:"))
        self.markers_view = QtWidgets.QTextEdit()
        self.markers_view.setReadOnly(True); self.markers_view.setMaximumHeight(220)
        ml.addWidget(self.markers_view)
        layout.addWidget(mk_group)
        layout.addStretch()
        return widget

    # ==================================================================
    # Bio Multimodal: container com 3 sub-abas (EMG, ECG, EoG)
    # ==================================================================
    def _build_bio_multimodal_tab(self):
        """Container das modalidades não-EEG: EMG / ECG / EoG.

        Cada sub-aba é independente — mostra cards/plots adequados ao tipo
        de bioelétrico. Só processa canais marcados com o respectivo tipo
        em 'Filtros e Canais → Tipo de sinal'.
        """
        outer = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(outer)
        v.setContentsMargins(4, 4, 4, 4); v.setSpacing(4)
        # Banner explicativo
        banner = QtWidgets.QLabel(
            "Modalidades bioelétricas — placa multimodal Bionica Lab. "
            "Configure o tipo de cada canal em <b>Filtros e Canais → Tipo de sinal</b>."
        )
        banner.setTextFormat(QtCore.Qt.TextFormat.RichText)
        banner.setStyleSheet(
            f"color: {COLORS['text_dim']}; padding: 4px 8px; "
            f"background: {COLORS['surface_alt']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: 3px;")
        banner.setWordWrap(True)
        v.addWidget(banner)

        # QTabWidget interno
        self.bio_tabs = QtWidgets.QTabWidget()
        self.bio_tabs.setDocumentMode(True)
        self.bio_tabs.setObjectName("bioTabs")
        # Sub-abas internas (cada uma envolvida em QScrollArea: conteúdo denso
        # com cards + plots + análise avançada — scroll vertical garante que
        # tudo seja acessível mesmo em telas menores).
        self.bio_tabs.addTab(self._wrap_scroll(self._build_emg_tab()),
                             "EMG · Músculos")
        self.bio_tabs.addTab(self._wrap_scroll(self._build_ecg_tab()),
                             "ECG · Coração")
        self.bio_tabs.addTab(self._wrap_scroll(self._build_eog_tab()),
                             "EoG · Olhos")
        self.bio_tabs.addTab(self._wrap_scroll(self._build_accelerometer_tab()),
                             "Acel · Movimento")
        v.addWidget(self.bio_tabs, stretch=1)
        return outer

    # ---- Tab: EMG / Músculos ----
    def _build_emg_tab(self):
        """Aba EMG — envelope, threshold e detecção de ativação muscular.

        A placa Bionica Lab é multimodal: cada canal pode ser EEG/EMG/ECG/EoG.
        Esta aba opera apenas nos canais marcados como EMG em
        config.channel_signal_types (configurável em 'Filtros e Canais').

        Cadeia de processamento EMG (por canal, em tempo real):
            sinal cru -> bandpass 20Hz-Nyquist + notch -> envelope
            (RMS móvel | |x|+LP | MAV) -> threshold com histerese -> LED ativo
        """
        widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(8, 8, 8, 8); outer.setSpacing(6)

        # === Cabeçalho informativo + métricas globais ===
        info_row = QtWidgets.QHBoxLayout()
        info = QtWidgets.QLabel(
            "Visualização EMG: envelope + threshold por canal. "
            "Configure quais canais são EMG em <b>Filtros e Canais → Tipo de sinal</b>."
        )
        info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        info.setStyleSheet(f"color: {COLORS['text_dim']};")
        info.setWordWrap(True)
        info_row.addWidget(info, stretch=3)
        self.emg_active_count_lbl = QtWidgets.QLabel("0 canais EMG ativos")
        self.emg_active_count_lbl.setStyleSheet(
            f"color: {SIGNAL_TYPE_COLORS['EMG']}; font-weight: bold; padding: 0 12px;"
            f"border: 1px solid {COLORS['border']}; border-radius: 3px;")
        info_row.addWidget(self.emg_active_count_lbl)
        outer.addLayout(info_row)

        # === Controles globais ===
        ctrl_group = QtWidgets.QGroupBox("Configuração do Envelope EMG")
        ctrl = QtWidgets.QHBoxLayout(ctrl_group)
        ctrl.setContentsMargins(8, 4, 8, 4)
        ctrl.addWidget(QtWidgets.QLabel("Método:"))
        self.emg_method_combo = QtWidgets.QComboBox()
        self.emg_method_combo.addItems(["RMS", "|x|+LP", "MAV"])
        self.emg_method_combo.setCurrentText(self.config.emg_envelope_method)
        self.emg_method_combo.setToolTip(
            "RMS    : Root Mean Square em janela móvel — padrão clínico.\n"
            "|x|+LP : Retificação seguida de Low-Pass (~10Hz).\n"
            "MAV    : Mean Absolute Value (média móvel do valor absoluto)."
        )
        self.emg_method_combo.currentTextChanged.connect(self._on_emg_settings_changed)
        ctrl.addWidget(self.emg_method_combo)
        ctrl.addSpacing(15)
        ctrl.addWidget(QtWidgets.QLabel("Janela (ms):"))
        self.emg_window_spin = QtWidgets.QSpinBox()
        self.emg_window_spin.setRange(10, 1000)
        self.emg_window_spin.setSingleStep(10)
        self.emg_window_spin.setValue(int(self.config.emg_envelope_window_ms))
        self.emg_window_spin.setToolTip("Janela em ms para cálculo do envelope (RMS/MAV).")
        self.emg_window_spin.valueChanged.connect(self._on_emg_settings_changed)
        ctrl.addWidget(self.emg_window_spin)
        ctrl.addSpacing(15)
        ctrl.addWidget(QtWidgets.QLabel("Threshold global:"))
        self.emg_global_thresh_spin = QtWidgets.QDoubleSpinBox()
        self.emg_global_thresh_spin.setRange(1.0, 5000.0)
        self.emg_global_thresh_spin.setDecimals(1)
        self.emg_global_thresh_spin.setSingleStep(5.0)
        self.emg_global_thresh_spin.setValue(50.0)
        self.emg_global_thresh_spin.setSuffix(" µV")
        ctrl.addWidget(self.emg_global_thresh_spin)
        btn_apply_all = QtWidgets.QPushButton("Aplicar a todos")
        btn_apply_all.clicked.connect(self._emg_apply_global_threshold)
        ctrl.addWidget(btn_apply_all)
        ctrl.addStretch()
        btn_reset_counters = QtWidgets.QPushButton("Reset contadores")
        btn_reset_counters.setToolTip("Zera contadores de ativação muscular por canal.")
        btn_reset_counters.clicked.connect(self._emg_reset_counters)
        ctrl.addWidget(btn_reset_counters)
        outer.addWidget(ctrl_group)

        # === Splitter: cards por canal (esquerda) + plot temporal (direita) ===
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # ----- Painel esquerdo: cards por canal -----
        cards_widget = QtWidgets.QWidget()
        cards_layout = QtWidgets.QGridLayout(cards_widget)
        cards_layout.setHorizontalSpacing(6); cards_layout.setVerticalSpacing(4)
        cards_layout.setContentsMargins(2, 2, 2, 2)

        hdr_titles = ["CH", "Tipo", "Eletrodo", "Envelope (µV)", "Pico", "Threshold (µV)", "Ativo", "Contagem"]
        for c, h in enumerate(hdr_titles):
            lbl = QtWidgets.QLabel(h)
            lbl.setStyleSheet(f"color: {COLORS['accent']}; font-weight: bold;")
            cards_layout.addWidget(lbl, 0, c)

        self.emg_rows = []  # lista de dicts por canal
        for ch in range(MAX_CHANNELS):
            row = ch + 1
            # CH
            ch_lbl = QtWidgets.QLabel(f"CH{ch+1}")
            ch_lbl.setStyleSheet(f"color: {CHANNEL_COLORS[ch]}; font-weight: bold;")
            cards_layout.addWidget(ch_lbl, row, 0)
            # Tipo (mostra tipo configurado; só "EMG" é processado por essa aba)
            type_lbl = QtWidgets.QLabel(self.config.channel_signal_types[ch])
            t_col = SIGNAL_TYPE_COLORS.get(self.config.channel_signal_types[ch], "#888")
            type_lbl.setStyleSheet(f"color: {t_col}; font-weight: bold;")
            cards_layout.addWidget(type_lbl, row, 1)
            # Eletrodo (do mapping)
            elec = (self.config.channel_mapping[ch]
                    if ch < len(self.config.channel_mapping) else f"E{ch+1}")
            elec_lbl = QtWidgets.QLabel(elec)
            elec_lbl.setStyleSheet(f"color: {COLORS['text_dim']};")
            cards_layout.addWidget(elec_lbl, row, 2)
            # Bar de envelope (QProgressBar)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 500)  # µV — ajustado dinamicamente
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setFormat("%v µV")
            bar.setStyleSheet(
                f"QProgressBar {{ border: 1px solid {COLORS['border']}; "
                f"  background: {COLORS['background']}; height: 18px; }} "
                f"QProgressBar::chunk {{ background-color: {SIGNAL_TYPE_COLORS['EMG']}; }}")
            cards_layout.addWidget(bar, row, 3)
            # Pico
            peak_lbl = QtWidgets.QLabel("0")
            peak_lbl.setStyleSheet(f"color: {COLORS['text_dim']}; font-family: {FONT_DATA_STACK};")
            peak_lbl.setMinimumWidth(50)
            cards_layout.addWidget(peak_lbl, row, 4)
            # Threshold (µV)
            th_spin = QtWidgets.QDoubleSpinBox()
            th_spin.setRange(1.0, 5000.0)
            th_spin.setDecimals(1)
            th_spin.setSingleStep(5.0)
            th_spin.setValue(self.config.emg_threshold_uV[ch] if ch < len(self.config.emg_threshold_uV) else 50.0)
            th_spin.setSuffix(" µV")
            th_spin.valueChanged.connect(
                lambda v, c=ch: self._emg_threshold_changed(c, v))
            cards_layout.addWidget(th_spin, row, 5)
            # LED de ativação
            led = QtWidgets.QLabel("●")
            led.setStyleSheet("color: #555; font-size: 18pt;")
            led.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            cards_layout.addWidget(led, row, 6)
            # Contagem de ativações
            cnt_lbl = QtWidgets.QLabel("0")
            cnt_lbl.setStyleSheet(f"color: {COLORS['text']}; font-family: {FONT_DATA_STACK};")
            cnt_lbl.setMinimumWidth(40)
            cards_layout.addWidget(cnt_lbl, row, 7)

            self.emg_rows.append({
                "type_lbl": type_lbl, "elec_lbl": elec_lbl,
                "bar": bar, "peak_lbl": peak_lbl, "th_spin": th_spin,
                "led": led, "count_lbl": cnt_lbl,
                "is_active": False, "activations": 0,
                "peak_envelope": 0.0,
            })

        cards_layout.setColumnStretch(3, 1)
        cards_widget.setLayout(cards_layout)
        cards_scroll = QtWidgets.QScrollArea()
        cards_scroll.setWidget(cards_widget)
        cards_scroll.setWidgetResizable(True)
        cards_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        split.addWidget(cards_scroll)

        # ----- Painel direito: plot temporal do envelope -----
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(2, 2, 2, 2)
        rl.addWidget(QtWidgets.QLabel("Envelope EMG — últimos 10 s"))
        self.emg_plot = pg.PlotWidget(enableMenu=False)
        self.emg_plot.showGrid(x=True, y=True, alpha=0.15)
        self.emg_plot.setLabel("left", "Envelope", units="µV")
        self.emg_plot.setLabel("bottom", "Tempo", units="s")
        self.emg_plot.addLegend(offset=(10, 10))
        self.emg_plot.setMenuEnabled(False)
        self.emg_curves = []
        for ch in range(MAX_CHANNELS):
            cur = self.emg_plot.plot(
                pen=pg.mkPen(CHANNEL_COLORS[ch], width=1.2),
                name=f"CH{ch+1}")
            cur.setVisible(False)
            self.emg_curves.append(cur)
        # Linhas de threshold (uma por canal — invisíveis até ser EMG)
        self.emg_threshold_lines = []
        for ch in range(MAX_CHANNELS):
            line = pg.InfiniteLine(
                pos=self.config.emg_threshold_uV[ch] if ch < len(self.config.emg_threshold_uV) else 50.0,
                angle=0,
                pen=pg.mkPen(CHANNEL_COLORS[ch],
                             style=QtCore.Qt.PenStyle.DashLine, width=1))
            line.setVisible(False)
            self.emg_plot.addItem(line)
            self.emg_threshold_lines.append(line)
        rl.addWidget(self.emg_plot)
        split.addWidget(right)
        split.setSizes([720, 600])
        outer.addWidget(split, stretch=2)

        # ===================================================================
        # SEÇÃO: Mapeamento Muscular + Análise Avançada
        # ===================================================================
        analysis_group = QtWidgets.QGroupBox(
            "Mapeamento Muscular e Análise Avançada (MVC, Fadiga, Co-contração)"
        )
        ag_l = QtWidgets.QVBoxLayout(analysis_group)
        ag_l.setContentsMargins(8, 6, 8, 8); ag_l.setSpacing(4)

        # Sub-controles superiores: indicadores globais
        gl = QtWidgets.QHBoxLayout()
        # Índice de fadiga global
        self.emg_fatigue_lbl = QtWidgets.QLabel("Fadiga: --")
        self.emg_fatigue_lbl.setStyleSheet(
            f"color: {COLORS['warning']}; font-weight: bold; padding: 4px 10px;"
            f"border: 1px solid {COLORS['border']}; border-radius: 4px;")
        self.emg_fatigue_lbl.setToolTip(
            "Índice de fadiga: queda da frequência mediana do espectro EMG "
            "ao longo da sessão. Queda >15% sugere fadiga muscular significativa."
        )
        gl.addWidget(self.emg_fatigue_lbl)
        # Co-contração
        self.emg_cocontraction_lbl = QtWidgets.QLabel("Co-contração: --")
        self.emg_cocontraction_lbl.setStyleSheet(
            f"color: {SIGNAL_TYPE_COLORS['EMG']}; font-weight: bold; padding: 4px 10px;"
            f"border: 1px solid {COLORS['border']}; border-radius: 4px;")
        self.emg_cocontraction_lbl.setToolTip(
            "Índice de co-contração entre canais antagonistas (agonista/antagonista).\n"
            "0% = só agonista ativo. 100% = ativação simultânea total.\n"
            "Alta co-contração pode indicar estabilização articular ou ineficiência."
        )
        gl.addWidget(self.emg_cocontraction_lbl)
        # Movimento dominante
        self.emg_movement_lbl = QtWidgets.QLabel("Movimento: --")
        self.emg_movement_lbl.setStyleSheet(
            f"color: {COLORS['accent']}; font-weight: bold; padding: 4px 10px;"
            f"border: 1px solid {COLORS['border']}; border-radius: 4px;")
        self.emg_movement_lbl.setToolTip(
            "Movimento dominante: para cada par agonista/antagonista mapeado, "
            "mostra qual está mais ativo (ex.: Flexão cotovelo / Extensão cotovelo)."
        )
        gl.addWidget(self.emg_movement_lbl)
        gl.addStretch()
        btn_reset_fatigue = QtWidgets.QPushButton("Reset fadiga (recalibra início)")
        btn_reset_fatigue.clicked.connect(self._emg_reset_fatigue_baseline)
        gl.addWidget(btn_reset_fatigue)
        ag_l.addLayout(gl)

        # Tabela de mapeamento muscular por canal
        # Layout: 8 colunas (CH + Músculo + Ação + %MVC + Calibrar MVC),
        # 2 blocos lado a lado (CH1-8 | CH9-16)
        muscle_grid = QtWidgets.QGridLayout()
        muscle_grid.setHorizontalSpacing(6); muscle_grid.setVerticalSpacing(2)
        for block in range(2):
            base_col = block * 5
            for i, h in enumerate(["CH", "Músculo", "Ação", "% MVC", "Calibrar MVC (3s)"]):
                lbl = QtWidgets.QLabel(h)
                lbl.setStyleSheet(f"color: {COLORS['accent']}; font-weight: bold;")
                muscle_grid.addWidget(lbl, 0, base_col + i)

        self.emg_muscle_combos = []
        self.emg_action_lbls   = []
        self.emg_mvc_pct_lbls  = []
        for ch in range(MAX_CHANNELS):
            block = ch // 8
            row = (ch % 8) + 1
            base_col = block * 5
            # CH
            ch_lbl = QtWidgets.QLabel(f"CH{ch+1}")
            ch_lbl.setStyleSheet(f"color: {CHANNEL_COLORS[ch]}; font-weight: bold;")
            muscle_grid.addWidget(ch_lbl, row, base_col + 0)
            # Combo de músculo
            mcb = QtWidgets.QComboBox()
            for muscle_name in COMMON_MUSCLES.keys():
                mcb.addItem(muscle_name)
            cur_muscle = (self.config.emg_channel_muscle[ch]
                          if ch < len(self.config.emg_channel_muscle) else "(não definido)")
            mcb.setCurrentText(cur_muscle)
            mcb.currentTextChanged.connect(
                lambda txt, c=ch: self._emg_muscle_changed(c, txt))
            muscle_grid.addWidget(mcb, row, base_col + 1)
            self.emg_muscle_combos.append(mcb)
            # Ação (auto-preenchido a partir do músculo)
            action_lbl = QtWidgets.QLabel(
                COMMON_MUSCLES.get(cur_muscle, {}).get("action", ""))
            action_lbl.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 9pt;")
            action_lbl.setMinimumWidth(110)
            muscle_grid.addWidget(action_lbl, row, base_col + 2)
            self.emg_action_lbls.append(action_lbl)
            # % MVC (atualizado em runtime)
            pct_lbl = QtWidgets.QLabel("--")
            pct_lbl.setStyleSheet(f"color: {COLORS['text']}; font-family: {FONT_DATA_STACK};")
            pct_lbl.setMinimumWidth(50)
            mvc_val = (self.config.emg_channel_mvc_uV[ch]
                       if ch < len(self.config.emg_channel_mvc_uV) else 0.0)
            pct_lbl.setToolTip(f"MVC calibrado: {mvc_val:.1f} µV (use 'Calibrar MVC' para refazer)")
            muscle_grid.addWidget(pct_lbl, row, base_col + 3)
            self.emg_mvc_pct_lbls.append(pct_lbl)
            # Botão calibrar MVC
            cal_btn = QtWidgets.QPushButton("Calibrar")
            cal_btn.setMaximumWidth(80)
            cal_btn.setToolTip(
                "Faz contração máxima por 3s para definir referência "
                "(MVC = Maximum Voluntary Contraction)."
            )
            cal_btn.clicked.connect(lambda _ck, c=ch: self._emg_mvc_calibrate(c))
            muscle_grid.addWidget(cal_btn, row, base_col + 4)

        muscle_grid.setColumnStretch(1, 1); muscle_grid.setColumnStretch(2, 1)
        muscle_grid.setColumnStretch(6, 1); muscle_grid.setColumnStretch(7, 1)
        ag_l.addLayout(muscle_grid)
        outer.addWidget(analysis_group, stretch=1)

        # ===================================================================
        # SEÇÃO: MNF / MDF temporal (rastreamento de fadiga muscular)
        # ===================================================================
        mnfdf_group = QtWidgets.QGroupBox(
            "Frequência Mediana (MDF) e Média (MNF) — Rastreamento de Fadiga"
        )
        mnfdf_l = QtWidgets.QVBoxLayout(mnfdf_group)
        mnfdf_l.setContentsMargins(8, 6, 8, 6)
        # Controles
        mnfdf_ctrl = QtWidgets.QHBoxLayout()
        mnfdf_ctrl.addWidget(QtWidgets.QLabel("Canal EMG:"))
        self.emg_mnfdf_channel = QtWidgets.QComboBox()
        for ch in range(MAX_CHANNELS):
            self.emg_mnfdf_channel.addItem(f"CH{ch+1}", ch)
        mnfdf_ctrl.addWidget(self.emg_mnfdf_channel)
        mnfdf_ctrl.addSpacing(15)
        self.emg_mnfdf_status_lbl = QtWidgets.QLabel(
            "MDF/MNF é a métrica clínica padrão (SENIAM): queda de 8-15% "
            "indica fadiga muscular."
        )
        self.emg_mnfdf_status_lbl.setStyleSheet(f"color: {COLORS['text_dim']};")
        mnfdf_ctrl.addWidget(self.emg_mnfdf_status_lbl, stretch=1)
        mnfdf_l.addLayout(mnfdf_ctrl)
        # Plot
        self.emg_mnfdf_plot = pg.PlotWidget(enableMenu=False)
        self.emg_mnfdf_plot.showGrid(x=True, y=True, alpha=0.15)
        self.emg_mnfdf_plot.setLabel("left", "Frequência", units="Hz")
        self.emg_mnfdf_plot.setLabel("bottom", "Janela (~1s cada)")
        self.emg_mnfdf_plot.setMenuEnabled(False)
        self.emg_mnfdf_plot.addLegend(offset=(10, 10))
        self.emg_mdf_curve = self.emg_mnfdf_plot.plot(
            pen=pg.mkPen(SIGNAL_TYPE_COLORS["EMG"], width=2), name="MDF (mediana)")
        self.emg_mnf_curve = self.emg_mnfdf_plot.plot(
            pen=pg.mkPen(COLORS["warning"], width=2,
                         style=QtCore.Qt.PenStyle.DashLine), name="MNF (média)")
        # Linha tracejada do baseline
        self.emg_mnfdf_baseline_line = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(COLORS["text_dim"], style=QtCore.Qt.PenStyle.DotLine))
        self.emg_mnfdf_plot.addItem(self.emg_mnfdf_baseline_line)
        self.emg_mnfdf_plot.setMinimumHeight(160)
        mnfdf_l.addWidget(self.emg_mnfdf_plot)
        outer.addWidget(mnfdf_group)

        # Buffer adicional para MNF (mean frequency)
        self._emg_mean_freq_history = {ch: [] for ch in range(MAX_CHANNELS)}

        # Buffer circular para envelope EMG (10 s)
        self.emg_envelope_buffer = np.zeros((MAX_CHANNELS, BUFFER_SIZE))
        # Histerese: True = canal já está em "ativo"; só sai quando cai abaixo de threshold*0.7
        self._emg_active_state = [False] * MAX_CHANNELS
        # MVC calibration state
        self._emg_mvc_calibrating = {}     # ch -> (start_time, data_list)
        # Fadiga: histórico da freq mediana por canal
        self._emg_median_freq_history = {ch: [] for ch in range(MAX_CHANNELS)}
        self._emg_fatigue_baseline = {ch: None for ch in range(MAX_CHANNELS)}

        # Aplica visibilidade inicial conforme channel_signal_types
        self._emg_refresh_channel_types()
        return widget

    # ---- Mapeamento muscular / MVC / Fadiga / Co-contração ----
    def _emg_muscle_changed(self, ch, muscle_name):
        if ch < 0 or ch >= MAX_CHANNELS: return
        if muscle_name not in COMMON_MUSCLES: return
        while len(self.config.emg_channel_muscle) <= ch:
            self.config.emg_channel_muscle.append("(não definido)")
        self.config.emg_channel_muscle[ch] = muscle_name
        # Atualiza label de ação
        info = COMMON_MUSCLES.get(muscle_name, {})
        if hasattr(self, "emg_action_lbls") and ch < len(self.emg_action_lbls):
            self.emg_action_lbls[ch].setText(info.get("action", ""))
        self.config.save()
        self._audit_event("emg_muscle_assignment", channel=ch+1, muscle=muscle_name)

    def _emg_mvc_calibrate(self, ch):
        """Inicia calibração MVC: 3 segundos de contração máxima."""
        if ch < 0 or ch >= MAX_CHANNELS: return
        self._emg_mvc_calibrating[ch] = (time.time(), [])
        muscle = self.config.emg_channel_muscle[ch] if ch < len(self.config.emg_channel_muscle) else "?"
        self._log(f"MVC CH{ch+1} ({muscle}): contraia ao máximo por 3s...")

    def _emg_reset_fatigue_baseline(self):
        """Reset baseline de fadiga para todos os canais."""
        self._emg_median_freq_history = {ch: [] for ch in range(MAX_CHANNELS)}
        self._emg_fatigue_baseline = {ch: None for ch in range(MAX_CHANNELS)}
        if hasattr(self, "emg_fatigue_lbl"):
            self.emg_fatigue_lbl.setText("Fadiga: -- (baseline resetado)")
        self._log("Baseline de fadiga EMG resetado.")

    # ---- Handlers EMG ----
    def _emg_threshold_changed(self, ch, val):
        """Usuário ajustou threshold de um canal."""
        if ch < 0 or ch >= MAX_CHANNELS: return
        # Estende lista se necessário
        while len(self.config.emg_threshold_uV) <= ch:
            self.config.emg_threshold_uV.append(50.0)
        self.config.emg_threshold_uV[ch] = float(val)
        if hasattr(self, "emg_threshold_lines"):
            self.emg_threshold_lines[ch].setPos(val)
        self.config.save()

    def _emg_apply_global_threshold(self):
        """Aplica o threshold global a todos os canais EMG."""
        if not hasattr(self, "emg_global_thresh_spin"): return
        val = self.emg_global_thresh_spin.value()
        for ch in range(MAX_CHANNELS):
            self.emg_rows[ch]["th_spin"].setValue(val)
        self._log(f"Threshold EMG aplicado a todos os canais: {val:.1f} µV")

    def _emg_reset_counters(self):
        """Zera contadores e picos."""
        for ch in range(MAX_CHANNELS):
            self.emg_rows[ch]["activations"] = 0
            self.emg_rows[ch]["peak_envelope"] = 0.0
            self.emg_rows[ch]["count_lbl"].setText("0")
            self.emg_rows[ch]["peak_lbl"].setText("0")

    def _on_emg_settings_changed(self, *_args):
        """Salva mudanças em método/janela do envelope."""
        if not hasattr(self, "emg_method_combo"): return
        self.config.emg_envelope_method = self.emg_method_combo.currentText()
        self.config.emg_envelope_window_ms = float(self.emg_window_spin.value())
        self.config.save()

    def _emg_refresh_channel_types(self):
        """Atualiza visibilidade dos canais conforme tipo configurado."""
        if not hasattr(self, "emg_rows"): return
        emg_count = 0
        for ch in range(MAX_CHANNELS):
            sig = (self.config.channel_signal_types[ch]
                   if ch < len(self.config.channel_signal_types) else "EEG")
            color = SIGNAL_TYPE_COLORS.get(sig, "#888")
            row = self.emg_rows[ch]
            row["type_lbl"].setText(sig)
            row["type_lbl"].setStyleSheet(f"color: {color}; font-weight: bold;")
            # Eletrodo atualizado
            elec = (self.config.channel_mapping[ch]
                    if ch < len(self.config.channel_mapping) else f"E{ch+1}")
            row["elec_lbl"].setText(elec)
            is_emg = (sig == "EMG")
            # Habilita/desabilita controles
            row["bar"].setEnabled(is_emg)
            row["th_spin"].setEnabled(is_emg)
            if is_emg:
                emg_count += 1
            else:
                # Apaga LED
                row["led"].setStyleSheet("color: #333; font-size: 18pt;")
                row["bar"].setValue(0)
            # Linhas e curvas no plot
            if hasattr(self, "emg_curves"):
                self.emg_curves[ch].setVisible(is_emg)
            if hasattr(self, "emg_threshold_lines"):
                self.emg_threshold_lines[ch].setVisible(is_emg)
        if hasattr(self, "emg_active_count_lbl"):
            self.emg_active_count_lbl.setText(f"{emg_count} canais EMG ativos")

    def _compute_emg_envelope(self, signal, fs, method, window_ms):
        """Calcula envelope EMG conforme método. signal: 1D ndarray."""
        if len(signal) == 0:
            return signal
        w = max(2, int(fs * window_ms / 1000.0))
        # Retifica primeiro
        rect = np.abs(signal)
        if method == "RMS":
            # RMS móvel = sqrt(media(x^2)) em janela w
            sq = signal * signal
            # Média móvel via convolução (boxcar)
            kernel = np.ones(w) / w
            ms = np.convolve(sq, kernel, mode="same")
            env = np.sqrt(np.maximum(ms, 0.0))
        elif method == "MAV":
            kernel = np.ones(w) / w
            env = np.convolve(rect, kernel, mode="same")
        else:  # "|x|+LP"
            # |x| seguido de média móvel (low-pass simples) com mesma janela
            kernel = np.ones(w) / w
            env = np.convolve(rect, kernel, mode="same")
        return env

    def _update_emg_view(self):
        """Atualiza envelope, LEDs, plot temporal, MVC, fadiga e co-contração."""
        if not hasattr(self, "emg_rows"):
            return
        data = self._ordered_buffer()
        n = data.shape[1]
        if n < 20:
            return
        method = self.config.emg_envelope_method
        window_ms = self.config.emg_envelope_window_ms
        t_axis = np.arange(n) / SAMPLE_RATE
        emg_count_active = 0
        # Armazena envelope atual por canal para análises agregadas
        ch_envelope_current = {}
        for ch in range(MAX_CHANNELS):
            sig = (self.config.channel_signal_types[ch]
                   if ch < len(self.config.channel_signal_types) else "EEG")
            if sig != "EMG":
                continue
            emg_count_active += 1
            try:
                env = self._compute_emg_envelope(data[ch], SAMPLE_RATE, method, window_ms)
            except Exception:
                continue
            row = self.emg_rows[ch]
            cur_val = float(env[-1]) if len(env) > 0 else 0.0
            peak_val = float(np.max(env)) if len(env) > 0 else 0.0
            ch_envelope_current[ch] = cur_val
            # Bar (escala automática: range = 2*threshold ou 100, o maior)
            th = row["th_spin"].value()
            bar_max = max(100, int(th * 3))
            row["bar"].setRange(0, bar_max)
            row["bar"].setValue(min(int(cur_val), bar_max))
            row["bar"].setFormat(f"{cur_val:.1f} µV")
            # Pico (acumula)
            if peak_val > row["peak_envelope"]:
                row["peak_envelope"] = peak_val
                row["peak_lbl"].setText(f"{peak_val:.1f}")
            # Histerese: ON quando > threshold, OFF quando < 0.7 * threshold
            was_active = self._emg_active_state[ch]
            if not was_active and cur_val > th:
                self._emg_active_state[ch] = True
                row["activations"] += 1
                row["count_lbl"].setText(str(row["activations"]))
                row["led"].setStyleSheet(
                    f"color: {SIGNAL_TYPE_COLORS['EMG']}; font-size: 18pt;")
            elif was_active and cur_val < 0.7 * th:
                self._emg_active_state[ch] = False
                row["led"].setStyleSheet("color: #555; font-size: 18pt;")
            # Curva temporal
            self.emg_curves[ch].setData(t_axis, env)

            # ---- MVC Calibration (3s contração máxima) ----
            if ch in self._emg_mvc_calibrating:
                start_t, samples = self._emg_mvc_calibrating[ch]
                samples.append(cur_val)
                if (time.time() - start_t) > 3.0:
                    # Finaliza: percentil 90 dos valores coletados
                    if samples:
                        mvc = float(np.percentile(samples, 90))
                        while len(self.config.emg_channel_mvc_uV) <= ch:
                            self.config.emg_channel_mvc_uV.append(0.0)
                        self.config.emg_channel_mvc_uV[ch] = mvc
                        self.config.save()
                        self._log(f"MVC CH{ch+1} calibrado: {mvc:.1f} µV")
                    self._emg_mvc_calibrating.pop(ch, None)

            # ---- % MVC ----
            if hasattr(self, "emg_mvc_pct_lbls") and ch < len(self.emg_mvc_pct_lbls):
                mvc = (self.config.emg_channel_mvc_uV[ch]
                       if ch < len(self.config.emg_channel_mvc_uV) else 0.0)
                if mvc > 0:
                    pct = (cur_val / mvc) * 100.0
                    color = (SIGNAL_TYPE_COLORS["EMG"] if pct < 80
                             else COLORS["warning"] if pct < 100
                             else COLORS["error"])
                    self.emg_mvc_pct_lbls[ch].setText(f"{pct:5.1f}%")
                    self.emg_mvc_pct_lbls[ch].setStyleSheet(
                        f"color: {color}; font-family: {FONT_DATA_STACK}; font-weight: bold;")
                else:
                    self.emg_mvc_pct_lbls[ch].setText("(cal)")

            # ---- Fadiga: mediana de freq do espectro ----
            # Atualiza só a cada ~1s (para não custar caro): usa fingerprint
            # do tamanho do histórico — atualiza quando samples_total múltiplo
            # de SAMPLE_RATE
            if getattr(self, "samples_total", 0) % SAMPLE_RATE == 0 and n > SAMPLE_RATE:
                try:
                    # FFT do último 1s
                    sig_win = data[ch, -SAMPLE_RATE:].astype(np.float64)
                    sig_win = sig_win - np.mean(sig_win)
                    spec = np.fft.rfft(sig_win * np.hanning(len(sig_win)))
                    psd  = (np.abs(spec) ** 2)
                    freqs = np.fft.rfftfreq(len(sig_win), 1.0 / SAMPLE_RATE)
                    total = float(np.sum(psd))
                    if total > 0:
                        cumsum = np.cumsum(psd)
                        med_idx = int(np.searchsorted(cumsum, total / 2.0))
                        med_freq = float(freqs[min(med_idx, len(freqs) - 1)])
                        # MNF = média ponderada pela PSD (Σ f·P / Σ P)
                        mean_freq = float(np.sum(freqs * psd) / total)
                        hist  = self._emg_median_freq_history.setdefault(ch, [])
                        mhist = self._emg_mean_freq_history.setdefault(ch, [])
                        hist.append(med_freq)
                        mhist.append(mean_freq)
                        if len(hist) > 600:   hist[:]  = hist[-600:]
                        if len(mhist) > 600:  mhist[:] = mhist[-600:]
                        if self._emg_fatigue_baseline.get(ch) is None and len(hist) >= 5:
                            self._emg_fatigue_baseline[ch] = float(np.mean(hist[:5]))
                except Exception:
                    pass

        # ---- Plot MNF/MDF temporal ----
        if hasattr(self, "emg_mdf_curve"):
            sel = self.emg_mnfdf_channel.currentData() if hasattr(self, "emg_mnfdf_channel") else 0
            if isinstance(sel, int) and sel >= 0:
                mdf = self._emg_median_freq_history.get(sel, [])
                mnf = self._emg_mean_freq_history.get(sel, [])
                if mdf:
                    self.emg_mdf_curve.setData(list(range(len(mdf))), mdf)
                if mnf:
                    self.emg_mnf_curve.setData(list(range(len(mnf))), mnf)
                base = self._emg_fatigue_baseline.get(sel)
                if base:
                    self.emg_mnfdf_baseline_line.setPos(base)
                # Status: queda atual
                if base and len(mdf) >= 3:
                    drop = (1.0 - float(np.mean(mdf[-3:])) / base) * 100.0
                    if hasattr(self, "emg_mnfdf_status_lbl"):
                        col = (SIGNAL_TYPE_COLORS["EMG"] if drop < 5
                               else COLORS["warning"] if drop < 15
                               else COLORS["error"])
                        self.emg_mnfdf_status_lbl.setText(
                            f"Queda MDF: {drop:+5.1f}% vs baseline ({base:.1f} Hz)"
                        )
                        self.emg_mnfdf_status_lbl.setStyleSheet(f"color: {col}; font-weight: bold;")

        # ---- Indicadores globais (fadiga + co-contração + movimento) ----
        if hasattr(self, "emg_fatigue_lbl"):
            # Fadiga média: queda % da freq mediana vs baseline
            drops = []
            for ch, hist in self._emg_median_freq_history.items():
                base = self._emg_fatigue_baseline.get(ch)
                if base and base > 0 and len(hist) >= 3:
                    recent = float(np.mean(hist[-3:]))
                    drop = (1.0 - recent / base) * 100.0
                    drops.append(drop)
            if drops:
                mean_drop = float(np.mean(drops))
                if   mean_drop < 5:  color = SIGNAL_TYPE_COLORS["EMG"]; state = "OK"
                elif mean_drop < 15: color = COLORS["warning"];          state = "leve"
                else:                color = COLORS["error"];            state = "alta"
                self.emg_fatigue_lbl.setText(f"Fadiga: {mean_drop:+5.1f}% ({state})")
                self.emg_fatigue_lbl.setStyleSheet(
                    f"color: {color}; font-weight: bold; padding: 4px 10px;"
                    f"border: 1px solid {COLORS['border']}; border-radius: 4px;")

        # ---- Co-contração e movimento dominante ----
        if hasattr(self, "emg_cocontraction_lbl"):
            cocontr_pairs = []
            dominant_moves = []
            for ch, env_val in ch_envelope_current.items():
                muscle = (self.config.emg_channel_muscle[ch]
                          if ch < len(self.config.emg_channel_muscle) else "(não definido)")
                if muscle == "(não definido)": continue
                info = COMMON_MUSCLES.get(muscle, {})
                antagonist = info.get("antagonist", "")
                if not antagonist: continue
                # Procura canal antagonista
                ant_ch = next((c for c in ch_envelope_current
                               if c < len(self.config.emg_channel_muscle)
                               and self.config.emg_channel_muscle[c] == antagonist), None)
                if ant_ch is None: continue
                ago_env = env_val
                ant_env = ch_envelope_current[ant_ch]
                if ago_env < 5 and ant_env < 5: continue  # ambos em repouso
                # Co-contração de Falconer-Winter: 2 * min / (a+b)
                cocontr = (2 * min(ago_env, ant_env)) / max(ago_env + ant_env, 1e-6) * 100.0
                cocontr_pairs.append(cocontr)
                # Movimento dominante: o agonista mais ativo
                if ago_env > ant_env * 1.2:
                    action = info.get("action", muscle)
                    dominant_moves.append(action)
                elif ant_env > ago_env * 1.2:
                    ant_info = COMMON_MUSCLES.get(antagonist, {})
                    action = ant_info.get("action", antagonist)
                    dominant_moves.append(action)
            if cocontr_pairs:
                mean_cc = float(np.mean(cocontr_pairs))
                self.emg_cocontraction_lbl.setText(f"Co-contração: {mean_cc:5.1f}%")
            else:
                self.emg_cocontraction_lbl.setText("Co-contração: -- (sem par)")
            if hasattr(self, "emg_movement_lbl"):
                if dominant_moves:
                    # Pega o mais comum
                    from collections import Counter as _C
                    top = _C(dominant_moves).most_common(1)[0][0]
                    self.emg_movement_lbl.setText(f"Movimento: {top}")
                else:
                    self.emg_movement_lbl.setText("Movimento: --")

    # ==================================================================
    # ECG / Coração — detecção de R-peak (Pan-Tompkins simplificado),
    # BPM, HRV (RMSSD, SDNN, pNN50) e Poincaré plot
    # ==================================================================
    def _build_ecg_tab(self):
        """Aba ECG — frequência cardíaca + HRV.

        Algoritmo:
            1. Bandpass 5-15 Hz (banda do QRS)
            2. Derivada (acentua bordas)
            3. Quadrado
            4. Média móvel (~150ms) = sinal MWA
            5. Threshold adaptativo (0.5 * média do MWA recente)
            6. Refratório: ignora picos < 200ms após o anterior
        """
        widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(8, 8, 8, 8); outer.setSpacing(6)

        # === Controles superiores ===
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Canal ECG:"))
        self.ecg_channel_combo = QtWidgets.QComboBox()
        # Popula com canais marcados ECG; se nenhum, adiciona "Auto (1º canal ECG)"
        self._populate_ecg_channel_combo()
        self.ecg_channel_combo.currentIndexChanged.connect(self._ecg_reset_history)
        ctrl.addWidget(self.ecg_channel_combo)
        ctrl.addSpacing(20)
        # BPM grande
        self.ecg_bpm_lbl = QtWidgets.QLabel("-- bpm")
        self.ecg_bpm_lbl.setStyleSheet(
            f"color: {SIGNAL_TYPE_COLORS['ECG']}; font-size: 26pt; font-weight: bold; "
            f"font-family: {FONT_DATA_STACK}; padding: 4px 16px;")
        self.ecg_bpm_lbl.setMinimumHeight(56)
        ctrl.addWidget(self.ecg_bpm_lbl)
        # LED de batimento (círculo pulsante — sem emoji)
        self.ecg_heartbeat_led = QtWidgets.QLabel("●")
        self.ecg_heartbeat_led.setStyleSheet(
            f"color: #555; font-size: 26pt; padding: 0 6px;")
        self.ecg_heartbeat_led.setToolTip("Pulsa a cada R-peak detectado")
        ctrl.addWidget(self.ecg_heartbeat_led)
        ctrl.addStretch()
        # Métricas HRV
        self.ecg_hrv_lbl = QtWidgets.QLabel("RMSSD: --  SDNN: --  pNN50: --")
        self.ecg_hrv_lbl.setStyleSheet(
            f"color: {COLORS['text_dim']}; font-family: {FONT_DATA_STACK};")
        ctrl.addWidget(self.ecg_hrv_lbl)
        btn_reset = QtWidgets.QPushButton("Reset")
        btn_reset.clicked.connect(self._ecg_reset_history)
        ctrl.addWidget(btn_reset)
        outer.addLayout(ctrl)

        # === Splitter: sinal cru + sinal MWA (esquerda) / Poincaré (direita) ===
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Painel esquerdo: 2 plots empilhados
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(2, 2, 2, 2)

        ll.addWidget(QtWidgets.QLabel("Sinal ECG (filtrado 5-15 Hz)"))
        self.ecg_raw_plot = pg.PlotWidget(enableMenu=False)
        self.ecg_raw_plot.showGrid(x=True, y=True, alpha=0.15)
        self.ecg_raw_plot.setLabel("left", "Amplitude", units="µV")
        self.ecg_raw_plot.setLabel("bottom", "Tempo", units="s")
        self.ecg_raw_plot.setMenuEnabled(False)
        self.ecg_raw_curve = self.ecg_raw_plot.plot(
            pen=pg.mkPen(SIGNAL_TYPE_COLORS["ECG"], width=1.4))
        # Marcadores de R-peak (ScatterPlotItem)
        self.ecg_rpeak_scatter = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush("#ffff00"),
            pen=pg.mkPen("#ffffff", width=1), symbol="t")
        self.ecg_raw_plot.addItem(self.ecg_rpeak_scatter)
        ll.addWidget(self.ecg_raw_plot)

        ll.addWidget(QtWidgets.QLabel("MWA (integral) + threshold"))
        self.ecg_mwa_plot = pg.PlotWidget(enableMenu=False)
        self.ecg_mwa_plot.showGrid(x=True, y=True, alpha=0.15)
        self.ecg_mwa_plot.setLabel("left", "MWA", units="µV²")
        self.ecg_mwa_plot.setLabel("bottom", "Tempo", units="s")
        self.ecg_mwa_plot.setMenuEnabled(False)
        self.ecg_mwa_curve = self.ecg_mwa_plot.plot(
            pen=pg.mkPen(COLORS["accent"], width=1.2))
        self.ecg_threshold_line = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen("#ff6677", style=QtCore.Qt.PenStyle.DashLine))
        self.ecg_mwa_plot.addItem(self.ecg_threshold_line)
        ll.addWidget(self.ecg_mwa_plot)

        # Sincronizar X dos dois plots
        self.ecg_mwa_plot.setXLink(self.ecg_raw_plot)

        split.addWidget(left)

        # Painel direito: tacograma + Poincaré
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(2, 2, 2, 2)

        rl.addWidget(QtWidgets.QLabel("Tacograma — intervalos RR (ms)"))
        self.ecg_tacho_plot = pg.PlotWidget(enableMenu=False)
        self.ecg_tacho_plot.showGrid(x=True, y=True, alpha=0.15)
        self.ecg_tacho_plot.setLabel("left", "RR", units="ms")
        self.ecg_tacho_plot.setLabel("bottom", "Batimento #")
        self.ecg_tacho_plot.setMenuEnabled(False)
        self.ecg_tacho_curve = self.ecg_tacho_plot.plot(
            pen=pg.mkPen(SIGNAL_TYPE_COLORS["ECG"], width=1.4),
            symbol="o", symbolSize=4,
            symbolBrush=SIGNAL_TYPE_COLORS["ECG"], symbolPen=None)
        rl.addWidget(self.ecg_tacho_plot)

        rl.addWidget(QtWidgets.QLabel("Poincaré — RR(n) × RR(n+1)"))
        self.ecg_poincare_plot = pg.PlotWidget(enableMenu=False)
        self.ecg_poincare_plot.showGrid(x=True, y=True, alpha=0.15)
        self.ecg_poincare_plot.setLabel("left",   "RR(n+1)", units="ms")
        self.ecg_poincare_plot.setLabel("bottom", "RR(n)",   units="ms")
        self.ecg_poincare_plot.setMenuEnabled(False)
        self.ecg_poincare_plot.setAspectLocked(True)
        # Linha identidade
        self.ecg_poincare_plot.addItem(pg.InfiniteLine(
            angle=45, pos=(0, 0),
            pen=pg.mkPen(COLORS["border"], style=QtCore.Qt.PenStyle.DashLine)))
        self.ecg_poincare_scatter = pg.ScatterPlotItem(
            size=6, brush=pg.mkBrush(SIGNAL_TYPE_COLORS["ECG"] + "aa"),
            pen=pg.mkPen(SIGNAL_TYPE_COLORS["ECG"]))
        self.ecg_poincare_plot.addItem(self.ecg_poincare_scatter)
        rl.addWidget(self.ecg_poincare_plot)

        split.addWidget(right)
        split.setSizes([720, 600])
        outer.addWidget(split, stretch=2)

        # ===================================================================
        # SEÇÃO: Zonas Karvonen + Arritmia + Recuperação + LF/HF
        # ===================================================================
        zone_group = QtWidgets.QGroupBox(
            "Zonas de Treino (Karvonen) + Detecção de Arritmia + Recuperação"
        )
        zg = QtWidgets.QHBoxLayout(zone_group)
        zg.setContentsMargins(8, 6, 8, 8); zg.setSpacing(10)

        # Idade (afeta HRmax)
        age_box = QtWidgets.QVBoxLayout()
        age_box.addWidget(QtWidgets.QLabel("Idade (HRmax = 220 - idade):"))
        age_row = QtWidgets.QHBoxLayout()
        self.ecg_age_spin = QtWidgets.QSpinBox()
        self.ecg_age_spin.setRange(5, 120); self.ecg_age_spin.setValue(25)
        self.ecg_age_spin.setSuffix(" anos")
        # Tenta pegar do voluntário ativo
        try:
            active = self.volunteers.get_active() if hasattr(self.volunteers, "get_active") else None
            if active and "idade" in active:
                idade = int(active.get("idade", 25))
                if 5 <= idade <= 120:
                    self.ecg_age_spin.setValue(idade)
        except Exception:
            pass
        self.ecg_age_spin.valueChanged.connect(self._ecg_recompute_zones)
        age_row.addWidget(self.ecg_age_spin)
        self.ecg_hrmax_lbl = QtWidgets.QLabel("HRmax: 195 bpm")
        self.ecg_hrmax_lbl.setStyleSheet(
            f"color: {COLORS['warning']}; font-family: {FONT_DATA_STACK}; font-weight: bold;")
        age_row.addWidget(self.ecg_hrmax_lbl)
        age_row.addStretch()
        age_box.addLayout(age_row)
        zg.addLayout(age_box)
        zg.addSpacing(15)

        # Zona atual
        zone_box = QtWidgets.QVBoxLayout()
        self.ecg_zone_lbl = QtWidgets.QLabel("Z1")
        self.ecg_zone_lbl.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 22pt; font-weight: bold;"
            f"padding: 6px 14px; border: 2px solid {COLORS['border']}; border-radius: 6px;")
        self.ecg_zone_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ecg_zone_lbl.setToolTip(
            "Zona Karvonen:\n"
            "  Z1 (50-60%): Recuperação ativa\n"
            "  Z2 (60-70%): Aeróbica leve - queima de gordura\n"
            "  Z3 (70-80%): Aeróbica - condicionamento\n"
            "  Z4 (80-90%): Limiar anaeróbico\n"
            "  Z5 (90-100%): Máximo / VO2max"
        )
        zone_box.addWidget(self.ecg_zone_lbl)
        zone_name_lbl = QtWidgets.QLabel("Zona de Treino")
        zone_name_lbl.setStyleSheet(f"color: {COLORS['text_dim']};")
        zone_name_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        zone_box.addWidget(zone_name_lbl)
        zg.addLayout(zone_box)
        zg.addSpacing(10)

        # Arritmia
        arr_box = QtWidgets.QVBoxLayout()
        self.ecg_arrhythmia_lbl = QtWidgets.QLabel("--")
        self.ecg_arrhythmia_lbl.setStyleSheet(
            f"color: {SIGNAL_TYPE_COLORS['ECG']}; font-size: 18pt; font-weight: bold;"
            f"padding: 6px 12px; border: 2px solid {COLORS['border']}; border-radius: 6px;")
        self.ecg_arrhythmia_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.ecg_arrhythmia_lbl.setMinimumWidth(160)
        self.ecg_arrhythmia_lbl.setToolTip(
            "Detecção simples baseada em coef. de variação dos intervalos RR:\n"
            "  CV<5%: Ritmo Regular\n  CV 5-15%: Variável (normal)\n"
            "  CV>15%: Irregular (verificar)\n"
            "Não é diagnóstico clínico — alerta heurístico."
        )
        arr_box.addWidget(self.ecg_arrhythmia_lbl)
        arr_lbl = QtWidgets.QLabel("Ritmo")
        arr_lbl.setStyleSheet(f"color: {COLORS['text_dim']};")
        arr_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        arr_box.addWidget(arr_lbl)
        zg.addLayout(arr_box)
        zg.addSpacing(10)

        # BPM pico / Recuperação
        rec_box = QtWidgets.QVBoxLayout()
        self.ecg_peak_bpm_lbl = QtWidgets.QLabel("Pico: -- bpm")
        self.ecg_peak_bpm_lbl.setStyleSheet(
            f"color: {COLORS['warning']}; font-family: {FONT_DATA_STACK}; font-weight: bold;")
        rec_box.addWidget(self.ecg_peak_bpm_lbl)
        self.ecg_recovery_lbl = QtWidgets.QLabel("Recuperação: --")
        self.ecg_recovery_lbl.setStyleSheet(
            f"color: {SIGNAL_TYPE_COLORS['ECG']}; font-family: {FONT_DATA_STACK};")
        rec_box.addWidget(self.ecg_recovery_lbl)
        # HRV freq (LF/HF)
        self.ecg_lfhf_lbl = QtWidgets.QLabel("LF/HF: --")
        self.ecg_lfhf_lbl.setStyleSheet(
            f"color: {COLORS['accent']}; font-family: {FONT_DATA_STACK};")
        self.ecg_lfhf_lbl.setToolTip(
            "Razão LF/HF (HRV espectral):\n"
            "LF = 0.04-0.15 Hz (simpático+parassimpático)\n"
            "HF = 0.15-0.4 Hz (parassimpático)\n"
            "Razão >2 sugere predominância simpática (estresse)."
        )
        rec_box.addWidget(self.ecg_lfhf_lbl)
        zg.addLayout(rec_box)
        zg.addStretch()

        outer.addWidget(zone_group)

        # ===================================================================
        # SEÇÃO: HRV não-linear (Kubios-style) — DFA, Entropia, Poincaré SD1/SD2
        # ===================================================================
        nlhrv_group = QtWidgets.QGroupBox(
            "HRV não-linear (Kubios-style) — DFA, Entropia, Poincaré"
        )
        nlhrv = QtWidgets.QHBoxLayout(nlhrv_group)
        nlhrv.setContentsMargins(8, 6, 8, 8); nlhrv.setSpacing(12)
        # Card DFA α1 / α2
        dfa_box = QtWidgets.QVBoxLayout()
        self.ecg_dfa_lbl = QtWidgets.QLabel("DFA α1: --  α2: --")
        self.ecg_dfa_lbl.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 14pt; font-weight: bold;"
            f"font-family: {FONT_DATA_STACK}; padding: 4px 8px;")
        self.ecg_dfa_lbl.setMinimumHeight(40)
        self.ecg_dfa_lbl.setToolTip(
            "Detrended Fluctuation Analysis:\n"
            "α1 (escalas curtas 4-16): saudável ~1.0; fadiga/risco <0.75\n"
            "α2 (escalas longas 16-64): saudável ~1.0\n"
            "Peng et al. (1995) — biomarcador de saúde cardíaca."
        )
        dfa_box.addWidget(self.ecg_dfa_lbl)
        dfa_lbl2 = QtWidgets.QLabel("Detrended Fluctuation Analysis")
        dfa_lbl2.setStyleSheet(f"color: {COLORS['text_dim']};")
        dfa_box.addWidget(dfa_lbl2)
        nlhrv.addLayout(dfa_box)

        # Card Entropia
        ent_box = QtWidgets.QVBoxLayout()
        self.ecg_entropy_lbl = QtWidgets.QLabel("ApEn: --  SampEn: --")
        self.ecg_entropy_lbl.setStyleSheet(
            f"color: {SIGNAL_TYPE_COLORS['ECG']}; font-size: 14pt; font-weight: bold;"
            f"font-family: {FONT_DATA_STACK}; padding: 4px 8px;")
        self.ecg_entropy_lbl.setMinimumHeight(40)
        self.ecg_entropy_lbl.setToolTip(
            "Entropia da série RR (complexidade/regularidade):\n"
            "  ApEn  (Pincus 1991): 0.5-1.5 é fisiológico\n"
            "  SampEn (Richman 2000): mais robusto, baseado em ApEn\n"
            "Valores baixos = ritmo regular; altos = caótico."
        )
        ent_box.addWidget(self.ecg_entropy_lbl)
        ent_lbl2 = QtWidgets.QLabel("Entropia da série RR")
        ent_lbl2.setStyleSheet(f"color: {COLORS['text_dim']};")
        ent_box.addWidget(ent_lbl2)
        nlhrv.addLayout(ent_box)

        # Card Poincaré SD1/SD2
        poin_box = QtWidgets.QVBoxLayout()
        self.ecg_poincare_sd_lbl = QtWidgets.QLabel("SD1: --  SD2: --  SD1/SD2: --")
        self.ecg_poincare_sd_lbl.setStyleSheet(
            f"color: {COLORS['warning']}; font-size: 14pt; font-weight: bold;"
            f"font-family: {FONT_DATA_STACK}; padding: 4px 8px;")
        self.ecg_poincare_sd_lbl.setMinimumHeight(40)
        self.ecg_poincare_sd_lbl.setToolTip(
            "Poincaré plot — descritores nuvem RR(n) × RR(n+1):\n"
            "  SD1: dispersão perpendicular (curto prazo, parassimpático)\n"
            "  SD2: dispersão paralela (longo prazo, simpático+parassimpático)\n"
            "  SD1/SD2: razão de balanço autonômico"
        )
        poin_box.addWidget(self.ecg_poincare_sd_lbl)
        poin_lbl = QtWidgets.QLabel("Poincaré (Brennan 2001)")
        poin_lbl.setStyleSheet(f"color: {COLORS['text_dim']};")
        poin_box.addWidget(poin_lbl)
        nlhrv.addLayout(poin_box)
        nlhrv.addStretch()

        outer.addWidget(nlhrv_group)

        # Estado interno
        self._ecg_rpeak_times = []    # lista de timestamps (s) dos picos R
        self._ecg_rr_intervals_ms = []  # ms entre picos
        self._ecg_last_beat_t = 0.0    # último tempo de batimento (para LED pulsante)
        self._ecg_threshold_adaptive = 0.0
        self._ecg_peak_bpm = 0.0       # BPM máximo observado na sessão
        self._ecg_peak_bpm_t = 0.0     # momento do pico
        self._ecg_recovery_60s = None  # diferença BPM 60s após o pico
        self._ecg_recompute_zones()
        return widget

    def _ecg_recompute_zones(self):
        """Recalcula HRmax (220 - idade) e exibe."""
        if not hasattr(self, "ecg_age_spin"): return
        age = self.ecg_age_spin.value()
        hrmax = 220 - age
        self._ecg_hrmax = hrmax
        if hasattr(self, "ecg_hrmax_lbl"):
            self.ecg_hrmax_lbl.setText(f"HRmax: {hrmax} bpm")

    def _populate_ecg_channel_combo(self):
        """(Re)popula combo com canais marcados ECG."""
        if not hasattr(self, "ecg_channel_combo"): return
        cur = self.ecg_channel_combo.currentText()
        self.ecg_channel_combo.blockSignals(True)
        self.ecg_channel_combo.clear()
        ecg_channels = [ch for ch in range(MAX_CHANNELS)
                        if ch < len(self.config.channel_signal_types)
                        and self.config.channel_signal_types[ch] == "ECG"]
        if not ecg_channels:
            self.ecg_channel_combo.addItem("(Nenhum canal ECG — configure em Filtros e Canais)", -1)
        else:
            for ch in ecg_channels:
                self.ecg_channel_combo.addItem(f"CH{ch+1}", ch)
        # Restaura seleção
        idx = self.ecg_channel_combo.findText(cur)
        if idx >= 0:
            self.ecg_channel_combo.setCurrentIndex(idx)
        self.ecg_channel_combo.blockSignals(False)

    def _ecg_reset_history(self):
        """Limpa histórico de picos R e RR."""
        self._ecg_rpeak_times = []
        self._ecg_rr_intervals_ms = []
        if hasattr(self, "ecg_bpm_lbl"):
            self.ecg_bpm_lbl.setText("-- bpm")
        if hasattr(self, "ecg_hrv_lbl"):
            self.ecg_hrv_lbl.setText("RMSSD: --  SDNN: --  pNN50: --")
        if hasattr(self, "ecg_rpeak_scatter"):
            self.ecg_rpeak_scatter.setData([], [])
        if hasattr(self, "ecg_tacho_curve"):
            self.ecg_tacho_curve.setData([], [])
        if hasattr(self, "ecg_poincare_scatter"):
            self.ecg_poincare_scatter.setData([], [])

    def _pan_tompkins_detect(self, signal, fs):
        """Pan-Tompkins simplificado. Retorna (indices_dos_picos_R, mwa_signal, threshold).

        Etapas: bandpass 5-15 Hz (aplicado aqui) -> derivada -> quadrado
        -> média móvel ~150ms -> threshold adaptativo + refratário 200ms.
        """
        n = len(signal)
        if n < int(fs * 0.5):
            return np.array([], dtype=int), np.zeros(n), 0.0
        signal = np.asarray(signal, dtype=float)
        # Bandpass 5-15 Hz do Pan-Tompkins — NAO assume filtragem previa (o filtro
        # global do app e banda-EEG); remove deriva de linha de base e ruido alto.
        try:
            ny = 0.5 * fs
            lo, hi = 5.0 / ny, min(15.0, ny * 0.95) / ny
            if 0 < lo < hi < 1 and np.all(np.isfinite(signal)):
                b, a = scipy_signal.butter(2, [lo, hi], btype="band")
                signal = scipy_signal.filtfilt(b, a, signal)
        except Exception:
            pass
        # Derivada
        diff = np.diff(signal, prepend=signal[0])
        # Quadrado
        sq = diff * diff
        # MWA (~150ms)
        w = max(2, int(fs * 0.15))
        kernel = np.ones(w) / w
        mwa = np.convolve(sq, kernel, mode="same")
        # Threshold adaptativo: 50% da média dos últimos 5s
        recent = mwa[-int(fs * 5):] if n >= int(fs * 5) else mwa
        peak_baseline = float(np.percentile(recent, 90))
        noise_baseline = float(np.percentile(recent, 30))
        threshold = noise_baseline + 0.5 * (peak_baseline - noise_baseline)
        threshold = max(threshold, 1e-6)
        # Detecção local de máximos acima do threshold com refratário 200ms
        refractory = int(fs * 0.2)
        peaks = []
        i = 1
        while i < n - 1:
            if mwa[i] > threshold and mwa[i] >= mwa[i-1] and mwa[i] >= mwa[i+1]:
                # encontra o máximo local em uma janela de ±25ms
                lo = max(0, i - int(fs * 0.025))
                hi = min(n, i + int(fs * 0.025) + 1)
                local_max = lo + int(np.argmax(mwa[lo:hi]))
                if not peaks or (local_max - peaks[-1]) >= refractory:
                    peaks.append(local_max)
                i = local_max + refractory
            else:
                i += 1
        return np.array(peaks, dtype=int), mwa, threshold

    def _update_ecg_view(self):
        """Atualiza ECG: roda Pan-Tompkins na janela atual, computa BPM e HRV."""
        if not hasattr(self, "ecg_raw_plot"): return
        # Canal selecionado
        ch = self.ecg_channel_combo.currentData() if hasattr(self, "ecg_channel_combo") else -1
        if ch is None or ch < 0:
            return
        if ch >= len(self.config.channel_signal_types) or \
                self.config.channel_signal_types[ch] != "ECG":
            return
        data = self._ordered_buffer()
        if data.shape[1] < int(SAMPLE_RATE * 1):
            return
        sig = data[ch].astype(np.float64)
        t_axis = np.arange(len(sig)) / SAMPLE_RATE
        # Atualiza curva crua (assumimos filtro bandpass já configurado no geral)
        self.ecg_raw_curve.setData(t_axis, sig)
        # Pan-Tompkins
        peaks, mwa, thr = self._pan_tompkins_detect(sig, SAMPLE_RATE)
        self.ecg_mwa_curve.setData(t_axis, mwa)
        self.ecg_threshold_line.setPos(thr)
        if len(peaks) > 0:
            self.ecg_rpeak_scatter.setData(
                t_axis[peaks].tolist(),
                sig[peaks].tolist())
            # RR intervals (ms)
            rr_samples = np.diff(peaks)
            rr_ms = rr_samples * 1000.0 / SAMPLE_RATE
            # Filtra outliers (faixa fisiológica 300-2000 ms ~ 30-200 bpm)
            rr_ms = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
            self._ecg_rr_intervals_ms = rr_ms.tolist()
            if len(rr_ms) > 0:
                # BPM = 60000/média RR
                mean_rr = float(np.mean(rr_ms[-10:]))  # média dos últimos 10
                bpm = 60000.0 / mean_rr if mean_rr > 0 else 0
                # Cor por faixa: <50 azul (baixa), 50-100 verde, 100-150 amarelo, >150 vermelho
                if bpm < 50:
                    color = "#88aaff"
                elif bpm < 100:
                    color = SIGNAL_TYPE_COLORS["ECG"]
                elif bpm < 150:
                    color = "#eebb33"
                else:
                    color = "#ee5566"
                self.ecg_bpm_lbl.setText(f"{bpm:.0f} bpm")
                self.ecg_bpm_lbl.setStyleSheet(
                    f"color: {color}; font-size: 26pt; font-weight: bold; "
                    f"font-family: {FONT_DATA_STACK}; padding: 4px 16px;")
                # HRV
                if len(rr_ms) >= 2:
                    diff = np.diff(rr_ms)
                    rmssd = float(np.sqrt(np.mean(diff * diff)))
                    sdnn  = float(np.std(rr_ms))
                    pnn50 = float(np.sum(np.abs(diff) > 50) / len(diff) * 100)
                    self.ecg_hrv_lbl.setText(
                        f"RMSSD: {rmssd:5.1f} ms  SDNN: {sdnn:5.1f} ms  pNN50: {pnn50:4.1f}%")
                # Tacograma + Poincaré
                self.ecg_tacho_curve.setData(np.arange(len(rr_ms)), rr_ms)
                if len(rr_ms) >= 2:
                    rr_n = rr_ms[:-1]; rr_np1 = rr_ms[1:]
                    self.ecg_poincare_scatter.setData(rr_n.tolist(), rr_np1.tolist())
                # LED pulsante: se o último pico está nos últimos 200ms da janela
                last_peak_t = float(t_axis[peaks[-1]])
                last_t = float(t_axis[-1])
                if (last_t - last_peak_t) < 0.25:
                    self.ecg_heartbeat_led.setStyleSheet(
                        f"color: {SIGNAL_TYPE_COLORS['ECG']}; font-size: 26pt; padding: 0 6px;")
                else:
                    self.ecg_heartbeat_led.setStyleSheet(
                        f"color: #555; font-size: 26pt; padding: 0 6px;")

                # ===== ZONA KARVONEN =====
                hrmax = getattr(self, "_ecg_hrmax", 195)
                # HR reserve usa HRrest = 70 como aproximação; uso direto % HRmax
                pct_hrmax = (bpm / hrmax) * 100.0 if hrmax > 0 else 0
                if   pct_hrmax < 60:  zone, zc = "Z1", "#88aaff"
                elif pct_hrmax < 70:  zone, zc = "Z2", SIGNAL_TYPE_COLORS["EoG"]
                elif pct_hrmax < 80:  zone, zc = "Z3", SIGNAL_TYPE_COLORS["EEG"]
                elif pct_hrmax < 90:  zone, zc = "Z4", COLORS["warning"]
                else:                 zone, zc = "Z5", COLORS["error"]
                if hasattr(self, "ecg_zone_lbl"):
                    self.ecg_zone_lbl.setText(f"{zone}  {pct_hrmax:.0f}%")
                    self.ecg_zone_lbl.setStyleSheet(
                        f"color: {zc}; font-size: 22pt; font-weight: bold;"
                        f"padding: 6px 14px; border: 2px solid {zc}; border-radius: 6px;")

                # ===== ARRITMIA (coef. variação RR) =====
                if hasattr(self, "ecg_arrhythmia_lbl") and len(rr_ms) >= 5:
                    cv = (float(np.std(rr_ms)) / max(float(np.mean(rr_ms)), 1e-6)) * 100.0
                    # outliers (RR > 1.5x mean ou < 0.5x mean)
                    mean_rr = float(np.mean(rr_ms))
                    outliers = np.sum((rr_ms > 1.5 * mean_rr) | (rr_ms < 0.5 * mean_rr))
                    if   cv < 5 and outliers == 0: status = "Regular";    sc = SIGNAL_TYPE_COLORS["ECG"]
                    elif cv < 15 and outliers <= 1: status = "Variável";  sc = COLORS["accent"]
                    elif cv < 25:                  status = "Irregular"; sc = COLORS["warning"]
                    else:                          status = "Verificar"; sc = COLORS["error"]
                    self.ecg_arrhythmia_lbl.setText(f"{status}\nCV={cv:.0f}%")
                    self.ecg_arrhythmia_lbl.setStyleSheet(
                        f"color: {sc}; font-size: 14pt; font-weight: bold;"
                        f"padding: 6px 12px; border: 2px solid {sc}; border-radius: 6px;")

                # ===== BPM PICO + RECUPERAÇÃO =====
                if bpm > self._ecg_peak_bpm:
                    self._ecg_peak_bpm = bpm
                    self._ecg_peak_bpm_t = time.time()
                    self._ecg_recovery_60s = None
                if hasattr(self, "ecg_peak_bpm_lbl"):
                    self.ecg_peak_bpm_lbl.setText(f"Pico: {self._ecg_peak_bpm:.0f} bpm")
                # Recuperação: queda de BPM 60s após o pico
                if self._ecg_peak_bpm > 0:
                    elapsed = time.time() - self._ecg_peak_bpm_t
                    if 50 < elapsed < 70 and self._ecg_recovery_60s is None:
                        self._ecg_recovery_60s = self._ecg_peak_bpm - bpm
                    if self._ecg_recovery_60s is not None:
                        if   self._ecg_recovery_60s >= 25: rec_text = "Excelente"; rc = SIGNAL_TYPE_COLORS["EEG"]
                        elif self._ecg_recovery_60s >= 15: rec_text = "Boa";       rc = SIGNAL_TYPE_COLORS["ECG"]
                        elif self._ecg_recovery_60s >= 5:  rec_text = "Lenta";     rc = COLORS["warning"]
                        else:                              rec_text = "Pobre";     rc = COLORS["error"]
                        if hasattr(self, "ecg_recovery_lbl"):
                            self.ecg_recovery_lbl.setText(
                                f"Recuperação 60s: -{self._ecg_recovery_60s:.0f} bpm ({rec_text})")
                            self.ecg_recovery_lbl.setStyleSheet(
                                f"color: {rc}; font-family: {FONT_DATA_STACK};")

                # ===== LF/HF (HRV espectral) =====
                # Precisa de >=20s de RR para resolver banda LF (0.04 Hz)
                if hasattr(self, "ecg_lfhf_lbl") and len(rr_ms) >= 20:
                    try:
                        # Interpola RR a 4 Hz (tempo regular)
                        t_rr = np.cumsum(rr_ms) / 1000.0  # tempo cumulativo em s
                        t_rr = t_rr - t_rr[0]
                        fs_interp = 4.0
                        n_interp = int(t_rr[-1] * fs_interp)
                        if n_interp > 16:
                            t_uniform = np.linspace(0, t_rr[-1], n_interp)
                            rr_interp = np.interp(t_uniform, t_rr, rr_ms)
                            rr_interp = rr_interp - np.mean(rr_interp)
                            # FFT
                            spec = np.fft.rfft(rr_interp * np.hanning(len(rr_interp)))
                            freqs = np.fft.rfftfreq(len(rr_interp), 1.0 / fs_interp)
                            psd = (np.abs(spec) ** 2) / (len(rr_interp) * fs_interp)
                            lf_mask = (freqs >= 0.04) & (freqs < 0.15)
                            hf_mask = (freqs >= 0.15) & (freqs < 0.4)
                            lf_p = float(_TRAPEZOID(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0
                            hf_p = float(_TRAPEZOID(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0
                            ratio = lf_p / max(hf_p, 1e-9)
                            # Validade: LF/HF confiavel exige ~2 min de RR (>=120
                            # batimentos). Em janela curta, sinaliza a limitacao.
                            caveat = "  ⚠ janela curta" if len(rr_ms) < 120 else ""
                            self.ecg_lfhf_lbl.setText(f"LF/HF: {ratio:.2f}{caveat}")
                    except Exception:
                        pass

                # ===== HRV NÃO-LINEAR (Kubios-style) =====
                if len(rr_ms) >= 32:
                    try:
                        rr_arr = np.asarray(rr_ms, dtype=np.float64)
                        # Poincaré SD1, SD2 (Brennan 2001)
                        diff = np.diff(rr_arr)
                        sd1 = float(np.std(diff) / np.sqrt(2.0))
                        sd2 = float(np.sqrt(max(0.0, 2 * np.var(rr_arr)
                                               - 0.5 * np.var(diff))))
                        sd_ratio = sd1 / max(sd2, 1e-9)
                        if hasattr(self, "ecg_poincare_sd_lbl"):
                            self.ecg_poincare_sd_lbl.setText(
                                f"SD1: {sd1:5.1f} ms  SD2: {sd2:5.1f} ms  "
                                f"SD1/SD2: {sd_ratio:.2f}"
                            )
                        # DFA α1 / α2 (Peng 1995)
                        a1 = self._compute_dfa(rr_arr, n_min=4,  n_max=16)
                        a2 = self._compute_dfa(rr_arr, n_min=16, n_max=min(64, len(rr_arr)//4))
                        if hasattr(self, "ecg_dfa_lbl"):
                            self.ecg_dfa_lbl.setText(
                                f"DFA α1: {a1:.2f}  α2: {a2:.2f}"
                            )
                        # ApEn / SampEn
                        apen   = self._compute_apen(rr_arr, m=2, r=0.2 * float(np.std(rr_arr)))
                        sampen = self._compute_sampen(rr_arr, m=2, r=0.2 * float(np.std(rr_arr)))
                        if hasattr(self, "ecg_entropy_lbl"):
                            self.ecg_entropy_lbl.setText(
                                f"ApEn: {apen:.3f}  SampEn: {sampen:.3f}"
                            )
                    except Exception:
                        pass
        else:
            self.ecg_rpeak_scatter.setData([], [])

    # ---- HRV não-linear (DFA, ApEn, SampEn) ----
    @staticmethod
    def _compute_dfa(x, n_min=4, n_max=64):
        """Detrended Fluctuation Analysis (Peng 1995).

        Retorna o expoente α (escala log-log de F(n) vs n).
        x: série temporal (RR em ms ou outro).
        """
        x = np.asarray(x, dtype=np.float64)
        N = len(x)
        if N < max(2 * n_max, 16):
            return float("nan")
        # Cumulative sum centralizada
        y = np.cumsum(x - np.mean(x))
        scales = np.unique(np.round(np.logspace(
            np.log10(max(n_min, 4)), np.log10(min(n_max, N // 4)), 8)).astype(int))
        F = []
        for n in scales:
            if n < 4 or n >= N: continue
            # Divide em blocos não-sobrepostos
            n_blocks = N // n
            if n_blocks < 2: continue
            errs = []
            for i in range(n_blocks):
                seg = y[i * n: (i + 1) * n]
                t = np.arange(n)
                # Ajuste linear (detrend)
                coef = np.polyfit(t, seg, 1)
                trend = np.polyval(coef, t)
                errs.append(np.mean((seg - trend) ** 2))
            if errs:
                F.append(np.sqrt(np.mean(errs)))
        if len(F) < 3:
            return float("nan")
        log_n = np.log10(scales[:len(F)])
        log_F = np.log10(F)
        # α = slope da regressão log-log
        slope = np.polyfit(log_n, log_F, 1)[0]
        return float(slope)

    @staticmethod
    def _compute_apen(x, m=2, r=None):
        """Approximate Entropy (Pincus 1991). Lower = more regular."""
        x = np.asarray(x, dtype=np.float64)
        N = len(x)
        if N < m + 1 or r is None or r <= 0:
            return float("nan")

        def _phi(m_val):
            count = 0
            total = 0
            for i in range(N - m_val + 1):
                template = x[i:i + m_val]
                hits = 0
                for j in range(N - m_val + 1):
                    if np.max(np.abs(x[j:j + m_val] - template)) <= r:
                        hits += 1
                if hits > 0:
                    count += np.log(hits / (N - m_val + 1))
                    total += 1
            return count / max(total, 1)

        return float(_phi(m) - _phi(m + 1))

    @staticmethod
    def _compute_sampen(x, m=2, r=None):
        """Sample Entropy (Richman 2000). Mais robusto que ApEn."""
        x = np.asarray(x, dtype=np.float64)
        N = len(x)
        if N < m + 1 or r is None or r <= 0:
            return float("nan")
        # Constrói templates de tamanho m e m+1
        def _count(m_val):
            B = 0
            for i in range(N - m_val):
                template = x[i:i + m_val]
                # excluindo self-match
                for j in range(N - m_val):
                    if i == j: continue
                    if np.max(np.abs(x[j:j + m_val] - template)) <= r:
                        B += 1
            return B
        Bm  = _count(m)
        Bmp = _count(m + 1)
        if Bm == 0 or Bmp == 0:
            return float("nan")
        return float(-np.log(Bmp / Bm))

    # ==================================================================
    # EoG / Olhos — eletrooculograma, detecção de blink + gaze direction
    # ==================================================================
    def _build_eog_tab(self):
        """Aba EoG — detecção de blink e estimativa de direção do olhar.

        Convenções:
            VEoG = canal vertical (cima/baixo) — eletrodo sobre/abaixo do olho
            HEoG = canal horizontal (esquerda/direita) — eletrodo lateral

        Detecção de blink:
            VEoG positivo > threshold por 100-400ms = piscada.
        Direção do olhar:
            VEoG > +T -> Cima
            VEoG < -T -> Baixo
            HEoG > +T -> Direita
            HEoG < -T -> Esquerda
            |V|<T e |H|<T -> Centro
        """
        widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(8, 8, 8, 8); outer.setSpacing(6)

        # === Controles ===
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Canal HEoG (horizontal):"))
        self.eog_h_combo = QtWidgets.QComboBox()
        ctrl.addWidget(self.eog_h_combo)
        ctrl.addSpacing(10)
        ctrl.addWidget(QtWidgets.QLabel("Canal VEoG (vertical):"))
        self.eog_v_combo = QtWidgets.QComboBox()
        ctrl.addWidget(self.eog_v_combo)
        ctrl.addSpacing(20)
        ctrl.addWidget(QtWidgets.QLabel("Threshold (µV):"))
        self.eog_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.eog_threshold_spin.setRange(5.0, 500.0)
        self.eog_threshold_spin.setValue(80.0)
        self.eog_threshold_spin.setSingleStep(5.0)
        self.eog_threshold_spin.setSuffix(" µV")
        ctrl.addWidget(self.eog_threshold_spin)
        ctrl.addStretch()
        btn_reset = QtWidgets.QPushButton("Reset contadores")
        btn_reset.clicked.connect(self._eog_reset)
        ctrl.addWidget(btn_reset)
        outer.addLayout(ctrl)
        self._populate_eog_channel_combos()
        self.eog_h_combo.currentIndexChanged.connect(self._eog_reset)
        self.eog_v_combo.currentIndexChanged.connect(self._eog_reset)

        # === Cards de métricas ===
        cards = QtWidgets.QHBoxLayout()
        # Direção atual
        self.eog_dir_lbl = QtWidgets.QLabel("Centro")
        self.eog_dir_lbl.setStyleSheet(
            f"color: {SIGNAL_TYPE_COLORS['EoG']}; font-size: 22pt; font-weight: bold; "
            f"padding: 10px 16px; border: 2px solid {COLORS['border']}; border-radius: 6px;")
        self.eog_dir_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.eog_dir_lbl.setMinimumSize(180, 60)
        cards.addWidget(self.eog_dir_lbl)
        # Blinks
        blink_box = QtWidgets.QVBoxLayout()
        self.eog_blink_count_lbl = QtWidgets.QLabel("0")
        self.eog_blink_count_lbl.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 24pt; font-weight: bold; padding: 4px;")
        self.eog_blink_count_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.eog_blink_count_lbl.setMinimumHeight(50)
        blink_box.addWidget(self.eog_blink_count_lbl)
        blink_lbl = QtWidgets.QLabel("piscadas detectadas")
        blink_lbl.setStyleSheet(f"color: {COLORS['text_dim']};")
        blink_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        blink_box.addWidget(blink_lbl)
        cards.addLayout(blink_box)
        cards.addSpacing(15)
        # Blink rate (pisca/min)
        rate_box = QtWidgets.QVBoxLayout()
        self.eog_blink_rate_lbl = QtWidgets.QLabel("--")
        self.eog_blink_rate_lbl.setStyleSheet(
            f"color: {COLORS['warning']}; font-size: 28pt; font-weight: bold;")
        self.eog_blink_rate_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        rate_box.addWidget(self.eog_blink_rate_lbl)
        rate_lbl = QtWidgets.QLabel("piscadas/min")
        rate_lbl.setStyleSheet(f"color: {COLORS['text_dim']};")
        rate_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        rate_box.addWidget(rate_lbl)
        cards.addLayout(rate_box)
        cards.addStretch()
        # Diagrama XY do olhar (dot que se move)
        self.eog_gaze_widget = _EogGazeWidget()
        self.eog_gaze_widget.setMinimumSize(220, 220)
        self.eog_gaze_widget.setMaximumSize(280, 280)
        cards.addWidget(self.eog_gaze_widget)
        outer.addLayout(cards)

        # === Plot dos 2 canais ===
        self.eog_plot = pg.PlotWidget(enableMenu=False)
        self.eog_plot.showGrid(x=True, y=True, alpha=0.15)
        self.eog_plot.setLabel("left", "Amplitude", units="µV")
        self.eog_plot.setLabel("bottom", "Tempo", units="s")
        self.eog_plot.setMenuEnabled(False)
        self.eog_plot.addLegend(offset=(10, 10))
        self.eog_h_curve = self.eog_plot.plot(
            pen=pg.mkPen("#66ddff", width=1.4), name="HEoG (horizontal)")
        self.eog_v_curve = self.eog_plot.plot(
            pen=pg.mkPen("#ffaadd", width=1.4), name="VEoG (vertical)")
        # Linhas de threshold (+/-)
        self.eog_thresh_pos_line = pg.InfiniteLine(
            pos=80, angle=0,
            pen=pg.mkPen(COLORS["border"], style=QtCore.Qt.PenStyle.DashLine))
        self.eog_thresh_neg_line = pg.InfiniteLine(
            pos=-80, angle=0,
            pen=pg.mkPen(COLORS["border"], style=QtCore.Qt.PenStyle.DashLine))
        self.eog_plot.addItem(self.eog_thresh_pos_line)
        self.eog_plot.addItem(self.eog_thresh_neg_line)
        # Scatter para marcar blinks detectados
        self.eog_blink_scatter = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush("#ffff00"),
            pen=pg.mkPen("#ffffff", width=1), symbol="t")
        self.eog_plot.addItem(self.eog_blink_scatter)
        # Scatter para sacadas
        self.eog_saccade_scatter = pg.ScatterPlotItem(
            size=8, brush=pg.mkBrush("#ff66ff"),
            pen=pg.mkPen("#ffffff", width=1), symbol="d")
        self.eog_plot.addItem(self.eog_saccade_scatter)
        outer.addWidget(self.eog_plot, stretch=2)
        # Listener de threshold
        self.eog_threshold_spin.valueChanged.connect(
            lambda v: (self.eog_thresh_pos_line.setPos(v),
                       self.eog_thresh_neg_line.setPos(-v)))

        # ===================================================================
        # SEÇÃO: Estado de Alerta + Sacadas + Fixação
        # ===================================================================
        alert_group = QtWidgets.QGroupBox(
            "Estado de Alerta (sonolência) + Sacadas + Fixação"
        )
        ag = QtWidgets.QHBoxLayout(alert_group)
        ag.setContentsMargins(8, 6, 8, 8); ag.setSpacing(12)

        # Estado de alerta (card grande)
        alert_box = QtWidgets.QVBoxLayout()
        self.eog_alert_state_lbl = QtWidgets.QLabel("--")
        self.eog_alert_state_lbl.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 22pt; font-weight: bold; "
            f"padding: 6px 12px; border: 2px solid {COLORS['border']}; border-radius: 6px;")
        self.eog_alert_state_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.eog_alert_state_lbl.setMinimumWidth(170)
        self.eog_alert_state_lbl.setToolTip(
            "Estado de alerta inferido a partir da taxa de piscadas:\n"
            "  ALERTA   : 8-18 piscadas/min (normal acordado)\n"
            "  SONOLENTO: <8 (olhos fechados) ou >22 (fadiga)\n"
            "  MICROSLEEP: olhos fechados por mais de 500 ms"
        )
        alert_box.addWidget(self.eog_alert_state_lbl)
        alert_lbl2 = QtWidgets.QLabel("Estado de Alerta")
        alert_lbl2.setStyleSheet(f"color: {COLORS['text_dim']};")
        alert_lbl2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        alert_box.addWidget(alert_lbl2)
        ag.addLayout(alert_box)

        # Sacadas
        sac_box = QtWidgets.QVBoxLayout()
        self.eog_saccade_count_lbl = QtWidgets.QLabel("0")
        self.eog_saccade_count_lbl.setStyleSheet(
            f"color: #ff66ff; font-size: 22pt; font-weight: bold;")
        self.eog_saccade_count_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        sac_box.addWidget(self.eog_saccade_count_lbl)
        sac_lbl = QtWidgets.QLabel("sacadas detectadas")
        sac_lbl.setStyleSheet(f"color: {COLORS['text_dim']};")
        sac_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        sac_box.addWidget(sac_lbl)
        ag.addLayout(sac_box)
        ag.addSpacing(10)

        # Fixação (% do tempo)
        fix_box = QtWidgets.QVBoxLayout()
        self.eog_fixation_pct_lbl = QtWidgets.QLabel("--")
        self.eog_fixation_pct_lbl.setStyleSheet(
            f"color: {SIGNAL_TYPE_COLORS['EoG']}; font-size: 22pt; font-weight: bold;")
        self.eog_fixation_pct_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        fix_box.addWidget(self.eog_fixation_pct_lbl)
        fix_lbl = QtWidgets.QLabel("% fixação (10 s)")
        fix_lbl.setStyleSheet(f"color: {COLORS['text_dim']};")
        fix_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        fix_box.addWidget(fix_lbl)
        ag.addLayout(fix_box)
        ag.addSpacing(10)

        # Duração média da piscada
        dur_box = QtWidgets.QVBoxLayout()
        self.eog_blink_dur_lbl = QtWidgets.QLabel("--")
        self.eog_blink_dur_lbl.setStyleSheet(
            f"color: {COLORS['warning']}; font-size: 22pt; font-weight: bold;")
        self.eog_blink_dur_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        dur_box.addWidget(self.eog_blink_dur_lbl)
        dur_lbl = QtWidgets.QLabel("ms (duração média)")
        dur_lbl.setStyleSheet(f"color: {COLORS['text_dim']};")
        dur_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        dur_box.addWidget(dur_lbl)
        ag.addLayout(dur_box)

        ag.addStretch()
        # Indicador de microsleep (alerta visual)
        self.eog_microsleep_lbl = QtWidgets.QLabel("●")
        self.eog_microsleep_lbl.setStyleSheet(
            f"color: #555; font-size: 30pt; padding: 0 10px;")
        self.eog_microsleep_lbl.setToolTip(
            "Aceso quando microsleep detectado (olhos fechados por > 500 ms)."
        )
        ag.addWidget(self.eog_microsleep_lbl)

        outer.addWidget(alert_group)

        # Estado interno
        self._eog_blink_count = 0
        self._eog_blink_times = []      # tempos absolutos
        self._eog_blink_durations = []  # ms
        self._eog_in_blink = False
        self._eog_blink_start_idx = 0
        self._eog_saccade_times = []
        self._eog_saccade_count = 0
        self._eog_last_microsleep_t = 0.0
        return widget

    def _populate_eog_channel_combos(self):
        """Popula combos H e V com canais marcados EoG."""
        if not hasattr(self, "eog_h_combo"): return
        for cb in (self.eog_h_combo, self.eog_v_combo):
            cb.blockSignals(True)
            cb.clear()
            eog_channels = [ch for ch in range(MAX_CHANNELS)
                            if ch < len(self.config.channel_signal_types)
                            and self.config.channel_signal_types[ch] == "EoG"]
            if not eog_channels:
                cb.addItem("(Nenhum canal EoG — configure em Filtros e Canais)", -1)
            else:
                for ch in eog_channels:
                    cb.addItem(f"CH{ch+1}", ch)
            cb.blockSignals(False)
        # Se houver pelo menos 2 canais EoG, default: 1º para H, 2º para V
        if self.eog_h_combo.count() >= 2 and self.eog_h_combo.itemData(0) is not None:
            self.eog_v_combo.setCurrentIndex(min(1, self.eog_v_combo.count()-1))

    def _eog_reset(self):
        self._eog_blink_count = 0
        self._eog_blink_times = []
        self._eog_blink_durations = []
        self._eog_in_blink = False
        self._eog_saccade_times = []
        self._eog_saccade_count = 0
        if hasattr(self, "eog_blink_count_lbl"):
            self.eog_blink_count_lbl.setText("0")
        if hasattr(self, "eog_blink_rate_lbl"):
            self.eog_blink_rate_lbl.setText("--")
        if hasattr(self, "eog_blink_scatter"):
            self.eog_blink_scatter.setData([], [])
        if hasattr(self, "eog_saccade_scatter"):
            self.eog_saccade_scatter.setData([], [])
        if hasattr(self, "eog_saccade_count_lbl"):
            self.eog_saccade_count_lbl.setText("0")
        if hasattr(self, "eog_alert_state_lbl"):
            self.eog_alert_state_lbl.setText("--")
        if hasattr(self, "eog_fixation_pct_lbl"):
            self.eog_fixation_pct_lbl.setText("--")
        if hasattr(self, "eog_blink_dur_lbl"):
            self.eog_blink_dur_lbl.setText("--")

    def _update_eog_view(self):
        """Atualiza canais H/V, detecta blinks e estima direção."""
        if not hasattr(self, "eog_plot"): return
        ch_h = self.eog_h_combo.currentData() if hasattr(self, "eog_h_combo") else -1
        ch_v = self.eog_v_combo.currentData() if hasattr(self, "eog_v_combo") else -1
        if ch_h is None or ch_v is None or ch_h < 0 or ch_v < 0:
            return
        data = self._ordered_buffer()
        if data.shape[1] < int(SAMPLE_RATE * 0.5): return
        sig_h = data[ch_h]; sig_v = data[ch_v]
        t_axis = np.arange(data.shape[1]) / SAMPLE_RATE
        self.eog_h_curve.setData(t_axis, sig_h)
        self.eog_v_curve.setData(t_axis, sig_v)

        th = self.eog_threshold_spin.value()
        # Direção atual: média dos últimos 250 ms, RELATIVA à baseline (mediana do
        # buffer) — remove a deriva lenta que travava a direção em Cima/Baixo.
        n_recent = int(SAMPLE_RATE * 0.25)
        h_base = float(np.median(sig_h)); v_base = float(np.median(sig_v))
        h_mean = float(np.mean(sig_h[-n_recent:])) - h_base
        v_mean = float(np.mean(sig_v[-n_recent:])) - v_base
        # Decisão direção
        if   v_mean >  th: direction = "Cima"
        elif v_mean < -th: direction = "Baixo"
        elif h_mean >  th: direction = "Direita"
        elif h_mean < -th: direction = "Esquerda"
        else:              direction = "Centro"
        self.eog_dir_lbl.setText(direction)
        # Atualiza gaze widget
        self.eog_gaze_widget.set_gaze(h_mean / max(th * 2, 1),
                                       v_mean / max(th * 2, 1))

        # Detecção de blink: pico de VEoG positivo > 1.5*th por 50-400ms
        # Faz busca local apenas na última janela curta para evitar reprocessar
        # picos antigos.
        n = len(sig_v)
        win = min(n, int(SAMPLE_RATE * 1.5))
        local = sig_v[-win:] - v_base          # relativo à baseline (sem deriva)
        peaks_h = []
        thr_blink = th * 1.5
        for i in range(1, len(local) - 1):
            if local[i] > thr_blink and local[i] >= local[i-1] and local[i] >= local[i+1]:
                peaks_h.append(i)
        # Adiciona apenas picos novos (distancia > 200ms do último)
        for p in peaks_h:
            t_p = float(t_axis[-win + p])
            if self._eog_blink_times and (t_p - self._eog_blink_times[-1]) < 0.2:
                continue
            if not self._eog_blink_times or (t_p - self._eog_blink_times[-1]) > 0.2:
                self._eog_blink_times.append(t_p)
                self._eog_blink_count += 1
        # Limita a lista (memória; evita a taxa fossilizar em sessão longa)
        if len(self._eog_blink_times) > 500:
            self._eog_blink_times = self._eog_blink_times[-500:]
        # Filtra apenas blinks nos últimos 10s (janela visível)
        last_t = float(t_axis[-1])
        visible_blinks = [t for t in self._eog_blink_times if (last_t - t) < BUFFER_SECONDS]
        # Atualiza scatter
        if visible_blinks:
            ys = [float(sig_v[int(t * SAMPLE_RATE)]) if int(t * SAMPLE_RATE) < n else 0
                  for t in visible_blinks]
            self.eog_blink_scatter.setData(visible_blinks, ys)
        # Atualiza UI
        self.eog_blink_count_lbl.setText(str(self._eog_blink_count))
        # Taxa: blinks dos últimos 60 s (extrapolado se for menos)
        rate = 0.0
        if len(self._eog_blink_times) >= 2:
            elapsed = max(1.0, last_t - self._eog_blink_times[0])
            rate = len(self._eog_blink_times) / elapsed * 60.0
            self.eog_blink_rate_lbl.setText(f"{rate:.0f}")

        # ===== DURAÇÃO DA PISCADA + MICROSLEEP =====
        # Para cada blink, estima duração via FWHM ao redor do pico
        try:
            durations = []
            for t_p in visible_blinks:
                idx = int(t_p * SAMPLE_RATE)
                if idx < 0 or idx >= n: continue
                peak_val = float(sig_v[idx])
                half = peak_val * 0.5
                # caminho para esquerda
                left = idx
                while left > 0 and sig_v[left] > half:
                    left -= 1
                right = idx
                while right < n - 1 and sig_v[right] > half:
                    right += 1
                dur_ms = (right - left) * 1000.0 / SAMPLE_RATE
                if 50 < dur_ms < 1500:
                    durations.append(dur_ms)
            if durations:
                mean_dur = float(np.mean(durations))
                self.eog_blink_dur_lbl.setText(f"{mean_dur:.0f}")
                # Microsleep: alguma piscada > 500 ms nos últimos 5 s
                long_blinks = [d for d in durations if d > 500]
                if long_blinks:
                    self._eog_last_microsleep_t = last_t
                # Mantém LED aceso por 2s
                if (last_t - self._eog_last_microsleep_t) < 2.0:
                    self.eog_microsleep_lbl.setStyleSheet(
                        f"color: {COLORS['error']}; font-size: 30pt; padding: 0 10px;")
                    self.eog_microsleep_lbl.setToolTip("MICROSLEEP detectado!")
                else:
                    self.eog_microsleep_lbl.setStyleSheet(
                        f"color: #555; font-size: 30pt; padding: 0 10px;")
        except Exception:
            pass

        # ===== ESTADO DE ALERTA =====
        if rate > 0:
            if   rate < 8:  state = "SONOLENTO";  color = COLORS["error"]
            elif rate > 22: state = "FADIGA";     color = COLORS["warning"]
            elif rate < 18: state = "ALERTA";     color = SIGNAL_TYPE_COLORS["EoG"]
            else:           state = "FADIGA?";    color = COLORS["warning"]
            self.eog_alert_state_lbl.setText(state)
            self.eog_alert_state_lbl.setStyleSheet(
                f"color: {color}; font-size: 22pt; font-weight: bold; "
                f"padding: 6px 12px; border: 2px solid {color}; border-radius: 6px;")

        # ===== SACADAS (derivada de HEoG > threshold) =====
        # Sacada = movimento rápido (velocidade > 30°/s)
        # Aproximamos via derivada de HEoG: dHEoG/dt > ~3*th por amostra
        try:
            n_h = len(sig_h)
            if n_h > SAMPLE_RATE:
                window_sac = sig_h[-SAMPLE_RATE:]  # último 1s
                dH = np.diff(window_sac)
                sac_thresh = th * 0.6  # µV por amostra
                # Encontra picos da derivada absoluta
                local_t0 = last_t - 1.0  # início do último 1s
                for i in range(1, len(dH) - 1):
                    if (abs(dH[i]) > sac_thresh and
                            abs(dH[i]) >= abs(dH[i-1]) and
                            abs(dH[i]) >= abs(dH[i+1])):
                        t_s = local_t0 + (i + 1) / SAMPLE_RATE
                        # refratório 100ms
                        if self._eog_saccade_times and (t_s - self._eog_saccade_times[-1]) < 0.1:
                            continue
                        self._eog_saccade_times.append(t_s)
                        self._eog_saccade_count += 1
                # Scatter visível dos últimos 10s
                vis_sac = [t for t in self._eog_saccade_times if (last_t - t) < BUFFER_SECONDS]
                if vis_sac:
                    ys = [float(sig_h[int(t * SAMPLE_RATE)])
                          if 0 <= int(t * SAMPLE_RATE) < n_h else 0
                          for t in vis_sac]
                    self.eog_saccade_scatter.setData(vis_sac, ys)
                self.eog_saccade_count_lbl.setText(str(self._eog_saccade_count))
        except Exception:
            pass

        # ===== FIXAÇÃO (% do tempo sem movimento) =====
        # Heurística: tempo em que |dH/dt| e |dV/dt| ambos < threshold/4
        try:
            if n > SAMPLE_RATE:
                window_fix = min(n, int(SAMPLE_RATE * BUFFER_SECONDS))
                wh = sig_h[-window_fix:]
                wv = sig_v[-window_fix:]
                dH = np.abs(np.diff(wh))
                dV = np.abs(np.diff(wv))
                no_move_mask = (dH < th * 0.15) & (dV < th * 0.15)
                pct = float(np.mean(no_move_mask)) * 100.0
                self.eog_fixation_pct_lbl.setText(f"{pct:5.1f}%")
        except Exception:
            pass

    # ==================================================================
    # Acelerômetro · Movimento — histórico completo da sessão + postura
    # ==================================================================
    def _build_accelerometer_tab(self):
        """Aba Acelerômetro — histórico de movimento da sessão inteira.

        Métricas:
            - X/Y/Z em g (gravidade) ao longo do tempo
            - Magnitude |g| = sqrt(x²+y²+z²)
            - Roll  = atan2(y, z) — inclinação lateral
            - Pitch = atan2(-x, sqrt(y²+z²)) — inclinação frontal
            - Activity Index = std móvel da magnitude (movimento detectado)
        """
        widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(8, 8, 8, 8); outer.setSpacing(6)

        # === Cards superiores ===
        cards = QtWidgets.QHBoxLayout()
        # Postura (roll/pitch)
        post_box = QtWidgets.QVBoxLayout()
        self.accel_posture_lbl = QtWidgets.QLabel("--")
        self.accel_posture_lbl.setStyleSheet(
            f"color: {COLORS['accent']}; font-size: 18pt; font-weight: bold; "
            f"padding: 6px 12px; border: 2px solid {COLORS['border']}; border-radius: 6px;")
        self.accel_posture_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.accel_posture_lbl.setMinimumWidth(220)
        self.accel_posture_lbl.setToolTip(
            "Postura inferida a partir da gravidade:\n"
            "  Vertical: -10° < pitch < 10°\n"
            "  Inclinado frente: pitch > 30°\n"
            "  Inclinado trás:  pitch < -30°\n"
            "  Lateral esq/dir: |roll| > 30°"
        )
        post_box.addWidget(self.accel_posture_lbl)
        post_lbl = QtWidgets.QLabel("Postura")
        post_lbl.setStyleSheet(f"color: {COLORS['text_dim']};")
        post_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        post_box.addWidget(post_lbl)
        cards.addLayout(post_box)
        cards.addSpacing(12)

        # Atividade
        act_box = QtWidgets.QVBoxLayout()
        self.accel_activity_lbl = QtWidgets.QLabel("--")
        self.accel_activity_lbl.setStyleSheet(
            f"color: {SIGNAL_TYPE_COLORS['EMG']}; font-size: 18pt; font-weight: bold; "
            f"padding: 6px 12px; border: 2px solid {COLORS['border']}; border-radius: 6px;")
        self.accel_activity_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.accel_activity_lbl.setMinimumWidth(150)
        self.accel_activity_lbl.setToolTip(
            "Índice de atividade:\n"
            "  PARADO:    std < 0.05 g\n"
            "  LEVE:      0.05-0.15 g\n"
            "  MODERADO:  0.15-0.5 g\n"
            "  INTENSO:   > 0.5 g"
        )
        act_box.addWidget(self.accel_activity_lbl)
        act_lbl2 = QtWidgets.QLabel("Atividade")
        act_lbl2.setStyleSheet(f"color: {COLORS['text_dim']};")
        act_lbl2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        act_box.addWidget(act_lbl2)
        cards.addLayout(act_box)
        cards.addSpacing(12)

        # Roll / Pitch numéricos
        rp_box = QtWidgets.QVBoxLayout()
        self.accel_roll_lbl = QtWidgets.QLabel("Roll:  +0.0°")
        self.accel_roll_lbl.setStyleSheet(
            f"color: {COLORS['text']}; font-family: {FONT_DATA_STACK};")
        rp_box.addWidget(self.accel_roll_lbl)
        self.accel_pitch_lbl = QtWidgets.QLabel("Pitch: +0.0°")
        self.accel_pitch_lbl.setStyleSheet(
            f"color: {COLORS['text']}; font-family: {FONT_DATA_STACK};")
        rp_box.addWidget(self.accel_pitch_lbl)
        self.accel_mag_lbl = QtWidgets.QLabel("|g|:    1.00")
        self.accel_mag_lbl.setStyleSheet(
            f"color: {COLORS['text']}; font-family: {FONT_DATA_STACK};")
        rp_box.addWidget(self.accel_mag_lbl)
        cards.addLayout(rp_box)
        cards.addStretch()

        # Total de eventos de movimento
        ev_box = QtWidgets.QVBoxLayout()
        self.accel_event_count_lbl = QtWidgets.QLabel("0")
        self.accel_event_count_lbl.setStyleSheet(
            f"color: {COLORS['warning']}; font-size: 22pt; font-weight: bold;")
        self.accel_event_count_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        ev_box.addWidget(self.accel_event_count_lbl)
        ev_lbl = QtWidgets.QLabel("eventos de movimento")
        ev_lbl.setStyleSheet(f"color: {COLORS['text_dim']};")
        ev_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        ev_box.addWidget(ev_lbl)
        cards.addLayout(ev_box)
        outer.addLayout(cards)

        # === Plot principal: histórico completo da sessão ===
        # 2 plots empilhados: (1) X/Y/Z (2) Magnitude + activity
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        # Plot 1: X/Y/Z históricos
        top_w = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(top_w); tl.setContentsMargins(2, 2, 2, 2)
        tl.addWidget(QtWidgets.QLabel("Acelerômetro X / Y / Z (sessão completa)"))
        self.accel_history_plot = pg.PlotWidget(enableMenu=False)
        self.accel_history_plot.showGrid(x=True, y=True, alpha=0.15)
        self.accel_history_plot.setLabel("left", "Aceleração", units="g")
        self.accel_history_plot.setLabel("bottom", "Tempo", units="s")
        self.accel_history_plot.addLegend(offset=(10, 10))
        self.accel_history_plot.setMenuEnabled(False)
        self.accel_x_curve = self.accel_history_plot.plot(
            pen=pg.mkPen("#ff6677", width=1.4), name="X (lateral)")
        self.accel_y_curve = self.accel_history_plot.plot(
            pen=pg.mkPen("#a3ff66", width=1.4), name="Y (frontal)")
        self.accel_z_curve = self.accel_history_plot.plot(
            pen=pg.mkPen("#66ddff", width=1.4), name="Z (vertical)")
        tl.addWidget(self.accel_history_plot)
        split.addWidget(top_w)

        # Plot 2: magnitude + activity
        bot_w = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bot_w); bl.setContentsMargins(2, 2, 2, 2)
        bl.addWidget(QtWidgets.QLabel("Magnitude |g| e Índice de Atividade"))
        self.accel_mag_plot = pg.PlotWidget(enableMenu=False)
        self.accel_mag_plot.showGrid(x=True, y=True, alpha=0.15)
        self.accel_mag_plot.setLabel("left", "Mag/Activity")
        self.accel_mag_plot.setLabel("bottom", "Tempo", units="s")
        self.accel_mag_plot.addLegend(offset=(10, 10))
        self.accel_mag_plot.setMenuEnabled(False)
        self.accel_mag_curve = self.accel_mag_plot.plot(
            pen=pg.mkPen(COLORS["accent"], width=1.4), name="|g|")
        self.accel_activity_curve = self.accel_mag_plot.plot(
            pen=pg.mkPen(SIGNAL_TYPE_COLORS["EMG"], width=1.4), name="Atividade")
        # Eventos de movimento (scatter)
        self.accel_event_scatter = pg.ScatterPlotItem(
            size=8, brush=pg.mkBrush("#ffaa55"),
            pen=pg.mkPen("#ffffff", width=1), symbol="t")
        self.accel_mag_plot.addItem(self.accel_event_scatter)
        # Linha activity threshold
        self.accel_activity_thr_line = pg.InfiniteLine(
            pos=0.15, angle=0,
            pen=pg.mkPen(COLORS["warning"], style=QtCore.Qt.PenStyle.DashLine))
        self.accel_mag_plot.addItem(self.accel_activity_thr_line)
        bl.addWidget(self.accel_mag_plot)
        split.addWidget(bot_w)
        split.setSizes([350, 250])
        outer.addWidget(split, stretch=1)

        # Estado interno: histórico downsampled da sessão (~10 Hz)
        self._accel_full_t = []  # tempo desde session_start (s)
        self._accel_full_x = []
        self._accel_full_y = []
        self._accel_full_z = []
        self._accel_full_mag = []
        self._accel_full_activity = []
        self._accel_last_log_t = 0.0
        self._accel_event_count = 0
        self._accel_event_times = []
        self._accel_in_event = False
        return widget

    def _update_accelerometer_view(self):
        """Atualiza histórico de acelerômetro, postura e atividade."""
        if not hasattr(self, "accel_history_plot"): return
        accel_data = self._ordered_accel()  # shape (3, N) últimos 10s
        if accel_data.shape[1] < 4: return
        # Valores atuais (média dos últimos 100 ms)
        n_recent = min(accel_data.shape[1], int(SAMPLE_RATE * 0.1))
        x_now = float(np.mean(accel_data[0, -n_recent:]))
        y_now = float(np.mean(accel_data[1, -n_recent:]))
        z_now = float(np.mean(accel_data[2, -n_recent:]))
        mag = float(np.sqrt(x_now**2 + y_now**2 + z_now**2))

        # Roll / Pitch (graus)
        import math
        roll  = math.degrees(math.atan2(y_now, max(abs(z_now), 1e-6) * (1 if z_now >= 0 else -1)))
        pitch = math.degrees(math.atan2(-x_now, math.sqrt(y_now**2 + z_now**2)))
        # Activity = std da magnitude no buffer recente
        n_w = min(accel_data.shape[1], int(SAMPLE_RATE * 1.0))
        mag_window = np.sqrt(accel_data[0, -n_w:]**2 +
                              accel_data[1, -n_w:]**2 +
                              accel_data[2, -n_w:]**2)
        activity = float(np.std(mag_window))

        # Cards
        self.accel_roll_lbl.setText(f"Roll:  {roll:+6.1f}°")
        self.accel_pitch_lbl.setText(f"Pitch: {pitch:+6.1f}°")
        self.accel_mag_lbl.setText(f"|g|:   {mag:5.2f}")

        # Postura
        if   abs(pitch) > 30:
            posture = "Inclinado " + ("frente" if pitch > 0 else "trás")
        elif abs(roll) > 30:
            posture = "Lateral "  + ("dir" if roll > 0 else "esq")
        elif mag < 0.5:
            posture = "Queda detectada!"
        else:
            posture = "Vertical (OK)"
        self.accel_posture_lbl.setText(posture)

        # Atividade
        if   activity < 0.05: act_text = "PARADO";    ac = SIGNAL_TYPE_COLORS["EoG"]
        elif activity < 0.15: act_text = "LEVE";      ac = SIGNAL_TYPE_COLORS["EEG"]
        elif activity < 0.5:  act_text = "MODERADO";  ac = COLORS["warning"]
        else:                 act_text = "INTENSO";   ac = COLORS["error"]
        self.accel_activity_lbl.setText(f"{act_text}\n{activity:.3f} g")
        self.accel_activity_lbl.setStyleSheet(
            f"color: {ac}; font-size: 16pt; font-weight: bold; "
            f"padding: 6px 12px; border: 2px solid {ac}; border-radius: 6px;")

        # Detecção de evento (entrada em estado MODERADO/INTENSO)
        was_in_event = self._accel_in_event
        now_in_event = activity > 0.15
        if now_in_event and not was_in_event:
            self._accel_event_count += 1
            self._accel_event_times.append(time.time())
            self.accel_event_count_lbl.setText(str(self._accel_event_count))
        self._accel_in_event = now_in_event

        # ===== Histórico downsampled (~10 Hz) =====
        now = time.time()
        if (now - self._accel_last_log_t) >= 0.1:
            self._accel_last_log_t = now
            t_rel = now - (self.session_start or now)
            self._accel_full_t.append(t_rel)
            self._accel_full_x.append(x_now)
            self._accel_full_y.append(y_now)
            self._accel_full_z.append(z_now)
            self._accel_full_mag.append(mag)
            self._accel_full_activity.append(activity)
            # Limita histórico a 1h (~36k pontos)
            max_pts = 36000
            if len(self._accel_full_t) > max_pts:
                for arr in (self._accel_full_t, self._accel_full_x,
                            self._accel_full_y, self._accel_full_z,
                            self._accel_full_mag, self._accel_full_activity):
                    del arr[: len(arr) - max_pts]
        # Atualiza plots (a cada chamada, dados já estão prontos)
        if self._accel_full_t:
            self.accel_x_curve.setData(self._accel_full_t, self._accel_full_x)
            self.accel_y_curve.setData(self._accel_full_t, self._accel_full_y)
            self.accel_z_curve.setData(self._accel_full_t, self._accel_full_z)
            self.accel_mag_curve.setData(self._accel_full_t, self._accel_full_mag)
            self.accel_activity_curve.setData(self._accel_full_t, self._accel_full_activity)
        # Scatter de eventos
        if self._accel_event_times and self.session_start:
            t_evs = [t - self.session_start for t in self._accel_event_times]
            ys = [0.3] * len(t_evs)
            self.accel_event_scatter.setData(t_evs, ys)

    # ---- Tab: Histórico ----
    def _build_history_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        info = QtWidgets.QLabel(
            f"Ultimos {BUFFER_SECONDS} segundos — todos os canais sobrepostos")
        info.setStyleSheet(f"color: {COLORS['text_dim']}; padding: 4px;")
        layout.addWidget(info)
        self.history_plot = pg.PlotWidget(enableMenu=False)
        self.history_plot.showGrid(x=True, y=True, alpha=0.15)
        self.history_plot.setLabel("left", "Amplitude", units="uV")
        self.history_plot.setLabel("bottom", "Tempo", units="s")
        self.history_plot.addLegend(offset=(10, 10))
        self.history_curves = []
        for ch in range(MAX_CHANNELS):
            curve = self.history_plot.plot(
                pen=pg.mkPen(CHANNEL_COLORS[ch], width=1.0), name=f"CH{ch + 1}")
            self.history_curves.append(curve)
        layout.addWidget(self.history_plot)
        return widget

    # ==================================================================
    # Expansão multi-step (8 → 16 → 24 → 32 → 40 → 48 → 56 → 64 canais)
    # ==================================================================
    def _snap_to_expansion_step(self, n):
        """Arredonda n para o passo válido mais próximo em EXPANSION_STEPS."""
        try:
            n = int(n)
        except (TypeError, ValueError):
            return BASE_CHANNELS
        n = max(BASE_CHANNELS, min(MAX_CHANNELS, n))
        if n in EXPANSION_STEPS:
            return n
        return min(EXPANSION_STEPS, key=lambda s: abs(s - n))

    def _on_expansion_combo_changed(self, idx):
        """Handler principal — chamado quando o usuário escolhe um novo passo."""
        if not hasattr(self, "expansion_combo"): return
        n = self.expansion_combo.itemData(idx)
        if n is None: return
        new_n = self._snap_to_expansion_step(int(n))
        self.daisy_enabled = (new_n > BASE_CHANNELS)
        self._set_num_channels(new_n)
        # Hardware: envia comando Cyton apenas para 8/16 (protocolo OpenBCI nativo)
        if self.serial_thread and self.serial_thread.isRunning() \
           and self.serial_thread.mode == "hardware":
            if new_n <= CYTON_MAX_CHANNELS:
                cmd = "C" if new_n > BASE_CHANNELS else "c"
                self.serial_thread.send_command(cmd)
                self._log(f"[Expansão] Comando '{cmd}' enviado à placa Cyton")
            else:
                self._log(
                    f"[Expansão] {new_n} canais excedem o protocolo Cyton+Daisy "
                    f"(máx {CYTON_MAX_CHANNELS}). Aplicando apenas no host — "
                    f"para hardware customizado (Bionica Lab) com mais módulos.",
                    error=False)

    def _on_expansion_toggled(self, checked):
        """Handler legacy (QCheckBox binário oculto) — mantido para compat."""
        self.daisy_enabled = bool(checked)
        new_n = CYTON_MAX_CHANNELS if self.daisy_enabled else BASE_CHANNELS
        self._set_num_channels(new_n)
        if self.serial_thread and self.serial_thread.isRunning() \
           and self.serial_thread.mode == "hardware":
            cmd = "C" if self.daisy_enabled else "c"
            self.serial_thread.send_command(cmd)
            self._log(f"[Expansão] Comando '{cmd}' enviado à placa")

    def _set_num_channels(self, n):
        n = self._snap_to_expansion_step(n)
        if n == self.num_channels:
            return
        self.num_channels = n
        self.expansion_label.setText(f"{n}ch")
        if n > BASE_CHANNELS:
            self.expansion_label.setStyleSheet(
                f"color: {COLORS['expansion']}; font-weight: bold; padding: 0 6px;"
                f"border: 1px solid {COLORS['expansion']}; border-radius: 3px;")
        else:
            self.expansion_label.setStyleSheet(
                f"color: {COLORS['text_dim']}; font-weight: bold; padding: 0 6px;"
                f"border: 1px solid {COLORS['border']}; border-radius: 3px;")
        # Sincroniza o combo principal sem reentrar no handler
        if hasattr(self, "expansion_combo"):
            try:
                target_idx = EXPANSION_STEPS.index(n)
            except ValueError:
                target_idx = 0
            if self.expansion_combo.currentIndex() != target_idx:
                self.expansion_combo.blockSignals(True)
                self.expansion_combo.setCurrentIndex(target_idx)
                self.expansion_combo.blockSignals(False)
        # Sincroniza o checkbox legacy (oculto) sem reentrar no handler
        want_checked = (n > BASE_CHANNELS)
        if hasattr(self, "expansion_toggle") and \
           self.expansion_toggle.isChecked() != want_checked:
            self.expansion_toggle.blockSignals(True)
            self.expansion_toggle.setChecked(want_checked)
            self.expansion_toggle.blockSignals(False)
        self.daisy_enabled = want_checked
        self._apply_channel_visibility()
        self.head_plot.set_num_channels(n)
        self.emg_widget.set_num_channels(n)
        # Propaga para combos de canal das abas multimodais (EMG / EoG / ECG / Focus)
        self._sync_multimodal_tabs_with_channels()
        self._log(f"Canais ativos: {n}")

    def _sync_multimodal_tabs_with_channels(self):
        """Propaga `num_channels` para TODAS as abas que listam canais.

        Quando o usuário escolhe na aba Conexão (ex.: 32 canais), todas as
        abas dependentes refletem isso automaticamente — sem precisar
        reconfigurar manualmente.

        Cobre:
            - Cards EMG (linhas + mapeamento muscular)
            - Combos ECG / EoG (filtra apenas canais marcados)
            - EMG Joystick (4 combos)
            - Focus / SSVEP (canal único EEG)
            - Aba Filtros e Canais (channel_type_combos, threshold inline,
              filter hint labels)
        """
        n = getattr(self, "num_channels", MAX_CHANNELS)
        # ---- EMG joystick — 4 combos por direção ----
        if hasattr(self, "_joy_axes"):
            try: self._joy_repopulate_combos()
            except Exception: pass
        # ---- ECG — combo de canal cardíaco ----
        if hasattr(self, "ecg_channel_combo"):
            try: self._populate_ecg_channel_combo()
            except Exception: pass
        # ---- EoG — combos H/V ----
        if hasattr(self, "eog_h_combo") and hasattr(self, "eog_v_combo"):
            try: self._populate_eog_channel_combos()
            except Exception: pass
        # ---- Focus — combo de canal único ----
        if hasattr(self, "focus_channel_combo"):
            try:
                prev = self.focus_channel_combo.currentData()
            except Exception:
                prev = None
            self.focus_channel_combo.blockSignals(True)
            self.focus_channel_combo.clear()
            for ch in range(n):
                elec = (self.config.channel_mapping[ch]
                        if ch < len(self.config.channel_mapping)
                        else f"E{ch+1}")
                self.focus_channel_combo.addItem(f"CH{ch+1} ({elec})", ch)
            if prev is not None and isinstance(prev, int) and prev < n:
                idx = self.focus_channel_combo.findData(prev)
                if idx >= 0:
                    self.focus_channel_combo.setCurrentIndex(idx)
            self.focus_channel_combo.blockSignals(False)
        # ---- Cards EMG (cada CH tem uma linha de widgets) ----
        if hasattr(self, "emg_rows"):
            for ch, row in enumerate(self.emg_rows):
                visible = (ch < n)
                for key in ("type_lbl", "elec_lbl", "bar", "peak_lbl",
                            "th_spin", "led", "count_lbl"):
                    w = row.get(key)
                    if w is not None:
                        w.setVisible(visible)
        # ---- Mapeamento muscular EMG (combos + ação + %MVC) ----
        if hasattr(self, "emg_muscle_combos"):
            for ch in range(min(len(self.emg_muscle_combos), MAX_CHANNELS)):
                self.emg_muscle_combos[ch].setVisible(ch < n)
        if hasattr(self, "emg_action_lbls"):
            for ch in range(min(len(self.emg_action_lbls), MAX_CHANNELS)):
                self.emg_action_lbls[ch].setVisible(ch < n)
        if hasattr(self, "emg_mvc_pct_lbls"):
            for ch in range(min(len(self.emg_mvc_pct_lbls), MAX_CHANNELS)):
                self.emg_mvc_pct_lbls[ch].setVisible(ch < n)
        # ---- Aba Filtros e Canais: combos de tipo + hint + threshold ----
        if hasattr(self, "channel_type_combos"):
            for ch in range(min(len(self.channel_type_combos), MAX_CHANNELS)):
                self.channel_type_combos[ch].setVisible(ch < n)
        if hasattr(self, "channel_filter_hint_lbls"):
            for ch in range(min(len(self.channel_filter_hint_lbls), MAX_CHANNELS)):
                self.channel_filter_hint_lbls[ch].setVisible(ch < n)
        if hasattr(self, "channel_emg_thresh_inline"):
            for ch in range(min(len(self.channel_emg_thresh_inline), MAX_CHANNELS)):
                self.channel_emg_thresh_inline[ch].setVisible(ch < n)

    def _apply_channel_visibility(self):
        """Mostra/esconde widgets dos canais 9-16 conforme num_channels atual."""
        n = self.num_channels
        for ch in range(MAX_CHANNELS):
            visible = (ch < n) and self.channel_active[ch]
            if hasattr(self, "channel_plots"):
                self.channel_plots[ch].setVisible(visible)
            if hasattr(self, "history_curves"):
                self.history_curves[ch].setVisible(visible)
            if hasattr(self, "channel_checks"):
                self.channel_checks[ch].setEnabled(ch < n)
                self.channel_checks[ch].setVisible(ch < n)
            # Stats table rows
            if hasattr(self, "stats_table"):
                self.stats_table.setRowHidden(ch, ch >= n)
            # Combos: habilita/desabilita itens 9-16
            for combo in (getattr(self, "analysis_channel", None),
                          getattr(self, "spec_channel", None),
                          getattr(self, "topo_channel_combo", None)):
                if combo is not None:
                    item = combo.model().item(ch)
                    if item is not None:
                        item.setEnabled(ch < n)
                    if ch < n and combo.currentIndex() >= n:
                        combo.setCurrentIndex(0)
            # Hardware grid rows
            if hasattr(self, "hw_row_widgets") and ch < len(self.hw_row_widgets):
                for w in self.hw_row_widgets[ch]:
                    w.setVisible(ch < n)
            # LEDs de qualidade
            if hasattr(self, "quality_leds") and ch < len(self.quality_leds):
                self.quality_leds[ch].setVisible(ch < n)

        # --- Atualiza altura mínima do GraphicsLayoutWidget ---
        # Sem isso, 16 canais ficariam espremidos em 700 px (zoom inviavel).
        # Com altura mínima por canal, ScrollArea aparece e zoom Y funciona limpo.
        if hasattr(self, "channels_layout") and hasattr(self, "_channels_per_plot_height"):
            n_visible = sum(1 for ch in range(MAX_CHANNELS)
                            if ch < n and self.channel_active[ch])
            total_h = max(1, n_visible) * self._channels_per_plot_height + 30
            self.channels_layout.setMinimumHeight(total_h)
            # Label "Tempo" so no ultimo canal visivel (limpo)
            last_visible = -1
            for ch in range(MAX_CHANNELS):
                if ch < n and self.channel_active[ch]:
                    last_visible = ch
            for ch in range(MAX_CHANNELS):
                if not hasattr(self, "channel_plots"): break
                show_xlabel = (ch == last_visible)
                bx = self.channel_plots[ch].getAxis("bottom")
                bx.setLabel("Tempo" if show_xlabel else "", units="s" if show_xlabel else "")
                bx.showLabel(show_xlabel)

        # Recalcula layout do modo Empilhado (ticks de canal)
        self._montage_last_nvis = -1   # força refresh no próximo _update_montage
        if hasattr(self, "montage_plot"):
            self._refresh_montage_layout()

    def _detect_expansion(self):
        """Tenta detectar automaticamente o número de canais ativo.

        Em Playback, conta as colunas do CSV: estrutura esperada é
        ``timestamp + N_canais + ax + ay + az [+ marker]`` (≥ 1 + N + 3).
        Snap-a para o passo mais próximo em EXPANSION_STEPS (8, 16, ..., 64).
        Em Hardware, envia '?' para a placa (apenas Cyton suporta esse
        protocolo; placas customizadas devem implementar resposta análoga).
        """
        mode_idx = self.mode_combo.currentIndex()
        if mode_idx == 2:  # Playback
            path = self.playback_path_edit.text().strip()
            if not path or not os.path.exists(path):
                self._log("Selecione um CSV de playback antes de detectar", error=True); return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    header = f.readline()
                ncols = header.count(",") + 1
                # canais inferidos = ncols - 1 (timestamp) - 3 (accel) - até 1 (marker)
                inferred = ncols - 1 - 3
                # Tenta sem marker e com marker
                candidates = [inferred, inferred - 1]
                detected = None
                for c in candidates:
                    if c in EXPANSION_STEPS:
                        detected = c; break
                if detected is None:
                    # snap ao passo mais próximo se estiver dentro do range
                    if BASE_CHANNELS <= inferred <= MAX_CHANNELS:
                        detected = self._snap_to_expansion_step(inferred)
                if detected is not None:
                    self._set_num_channels(detected)
                    extra = detected - BASE_CHANNELS
                    desc = ("placa base" if detected == BASE_CHANNELS
                            else f"base + {extra//8} módulo(s) de expansão")
                    self._log(f"✓ Auto-detecção: CSV tem {ncols} colunas → "
                              f"modo {detected} canais ({desc})")
                else:
                    self._log(f"Auto-detecção: CSV não reconhecido ({ncols} colunas, "
                              f"~{inferred} canais inferidos)", error=True)
            except Exception as exc:
                self._log(f"Erro lendo cabeçalho do CSV: {exc}", error=True)
        elif mode_idx == 1:  # Simulação
            self._log("Modo simulação: escolha o número de canais no seletor "
                      "(8 / 16 / 24 / 32 / 40 / 48 / 56 / 64)")
        else:  # Hardware
            if not (self.serial_thread and self.serial_thread.isRunning()):
                self._log("Conecte primeiro em modo Hardware para detectar via dispositivo", error=True); return
            self.serial_thread.send_command("?")  # pede status (Cyton)
            self._log("[Expansão] Comando '?' enviado — verifique resposta no log do dispositivo. "
                      "Em seguida, ajuste manualmente o seletor de canais conforme detectado.")

    # ==================================================================
    # Handlers
    # ==================================================================
    def _refresh_ports(self):
        self.port_combo.clear()
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            self.port_combo.addItem("(nenhuma porta detectada)", None)
        else:
            for p in ports:
                self.port_combo.addItem(f"{p.device} — {p.description}", p.device)
        self._log(f"Portas atualizadas — {len(ports)} encontrada(s)")

    # ==================================================================
    # Bluetooth — scan BLE async + ajuda de pareamento
    # ==================================================================
    def _bt_scan_start(self):
        """Inicia scan BLE em thread separada (não bloqueia UI)."""
        if not hasattr(self, "bt_devices_table"): return
        # Se já tem scan em andamento, avisa
        if self._bt_scanner_thread and self._bt_scanner_thread.isRunning():
            self.bt_status_lbl.setText("Já tem um scan em andamento...")
            return
        if not HAS_BLEAK:
            QtWidgets.QMessageBox.information(
                self, "Bluetooth indisponível",
                "Para varrer dispositivos BLE, instale a biblioteca:\n\n"
                "    pip install bleak\n\n"
                "Para dispositivos Bluetooth Clássico (HC-05, módulos seriais), "
                "pareie diretamente pelo Windows (Configurações → Bluetooth) — "
                "uma porta COM virtual será criada e aparecerá na lista de portas."
            )
            self.bt_status_lbl.setText("bleak não instalado — pip install bleak")
            return
        self.bt_devices_table.setRowCount(0)
        self.bt_scan_btn.setEnabled(False)
        self.bt_status_lbl.setText("Procurando dispositivos BLE (8 s)...")
        self.bt_status_lbl.setStyleSheet(f"color: {COLORS['warning']};")
        # Cria e lança thread
        self._bt_scanner_thread = _BluetoothScanThread(duration_s=8.0, parent=self)
        self._bt_scanner_thread.scan_done.connect(self._bt_on_scan_done)
        self._bt_scanner_thread.scan_failed.connect(self._bt_on_scan_failed)
        self._bt_scanner_thread.finished.connect(
            lambda: self.bt_scan_btn.setEnabled(True))
        self._bt_scanner_thread.start()
        self._audit_event("bluetooth_scan_started", duration_s=8)

    def _bt_on_scan_done(self, devices):
        """Recebe lista de devices e popula tabela."""
        if not hasattr(self, "bt_devices_table"): return
        # Ordena por RSSI (mais forte primeiro)
        devices_sorted = sorted(
            devices,
            key=lambda d: (-d.get("rssi", -999) if d.get("rssi") is not None else 999)
        )
        self.bt_devices_table.setRowCount(len(devices_sorted))
        for i, dev in enumerate(devices_sorted):
            name_item = QtWidgets.QTableWidgetItem(dev.get("name", ""))
            addr_item = QtWidgets.QTableWidgetItem(dev.get("address", ""))
            rssi = dev.get("rssi")
            rssi_item = QtWidgets.QTableWidgetItem(f"{rssi}" if rssi is not None else "-")
            # Cor por força de sinal
            if rssi is not None:
                if rssi > -60:   rssi_item.setForeground(QtGui.QColor(SIGNAL_TYPE_COLORS["EEG"]))
                elif rssi > -75: rssi_item.setForeground(QtGui.QColor(COLORS["warning"]))
                else:            rssi_item.setForeground(QtGui.QColor(COLORS["error"]))
            type_item = QtWidgets.QTableWidgetItem(dev.get("type", "BLE"))
            self.bt_devices_table.setItem(i, 0, name_item)
            self.bt_devices_table.setItem(i, 1, addr_item)
            self.bt_devices_table.setItem(i, 2, rssi_item)
            self.bt_devices_table.setItem(i, 3, type_item)
        self.bt_status_lbl.setText(f"{len(devices_sorted)} dispositivo(s) encontrado(s).")
        self.bt_status_lbl.setStyleSheet(f"color: {SIGNAL_TYPE_COLORS['EEG']};")
        self._log(f"Bluetooth scan concluído: {len(devices_sorted)} dispositivos")
        self._audit_event("bluetooth_scan_done", count=len(devices_sorted))

    def _bt_on_scan_failed(self, error_msg):
        self.bt_status_lbl.setText(f"Falha: {error_msg}")
        self.bt_status_lbl.setStyleSheet(f"color: {COLORS['error']};")
        self._log(f"Bluetooth scan falhou: {error_msg}", error=True)

    def _bt_show_pair_help(self):
        """Mostra instruções de pareamento Bluetooth no Windows."""
        QtWidgets.QMessageBox.information(
            self, "Como parear via Bluetooth (Windows)",
            "<b>Para placas com BT Serial (HC-05 / HC-06 / Cyton-BT):</b><br><br>"
            "1. Abra <b>Configurações → Dispositivos → Bluetooth e outros</b><br>"
            "2. Clique em <b>Adicionar Bluetooth</b> → Bluetooth<br>"
            "3. Selecione sua placa (PIN comum: <code>1234</code> ou <code>0000</code>)<br>"
            "4. Após parear, abra <b>Mais opções de Bluetooth → COM Ports</b><br>"
            "5. Anote a porta COM <b>outgoing</b> (ex.: COM7)<br>"
            "6. Volte aqui, clique <b>Atualizar</b> na lista de portas, "
            "selecione a COMx e conecte normalmente.<br><br>"
            "<b>Para placas BLE puras (Ganglion):</b> use o scan acima — "
            "a conexão BLE direta exige firmware específico que sua placa "
            "Bionica Lab deve implementar (GATT services com characteristic "
            "para streaming de amostras EEG)."
        )

    def _on_mode_changed(self, idx):
        is_playback = (idx == 2)
        self.playback_path_edit.setEnabled(is_playback)
        self.playback_browse_btn.setEnabled(is_playback)
        self.playback_progress.setVisible(is_playback)
        # Watermark "SIMULAÇÃO / PLAYBACK"
        if hasattr(self, "mode_combo"):
            self._update_simulation_overlay(self.mode_combo.currentText())

    def _select_playback_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Selecionar CSV para playback", self.config.save_directory,
            "CSV (*.csv);;Todos (*)")
        if path:
            self.playback_path_edit.setText(path)
            # Auto-detecta canais ao escolher o arquivo
            self._detect_expansion()

    def _detect_bad_channels(self):
        """Detecta canais ruins via 3 critérios:
        1) Variância > 5× mediana das variâncias (saturação/ruído)
        2) Variância < 0.05× mediana (canal "morto" / desconectado)
        3) Correlação média com vizinhos < 0.2 (eletrodo solto)
        """
        if not hasattr(self, "_ordered_buffer"):
            return
        data = self._ordered_buffer()
        n_total = data.shape[1]
        if n_total < SAMPLE_RATE * 2:
            self.bad_detect_status.setText("Coletar pelo menos 2s de sinal antes.")
            self.bad_detect_status.setStyleSheet(f"color: {COLORS['warning']};")
            return
        n = getattr(self, "num_channels", MAX_CHANNELS)
        bad = []
        reasons = {}
        # Variâncias por canal
        var_per_ch = np.var(data[:n], axis=1)
        med_var = float(np.median(var_per_ch[var_per_ch > 0])) if (var_per_ch > 0).any() else 1.0
        # Correlação com vizinhos
        for ch in range(n):
            r = []
            # Variância anômala
            if var_per_ch[ch] > 5 * med_var:
                bad.append(ch); reasons[ch] = f"var alta ({var_per_ch[ch]:.0f})"; continue
            if var_per_ch[ch] < 0.05 * med_var:
                bad.append(ch); reasons[ch] = f"var baixa ({var_per_ch[ch]:.1f})"; continue
            # Correlação com até 4 vizinhos
            for j in range(max(0, ch-2), min(n, ch+3)):
                if j == ch: continue
                s_ch = data[ch] - np.mean(data[ch])
                s_j  = data[j]  - np.mean(data[j])
                denom = np.std(data[ch]) * np.std(data[j])
                if denom > 0:
                    r.append(abs(float(np.mean(s_ch * s_j) / denom)))
            if r and float(np.mean(r)) < 0.2:
                bad.append(ch); reasons[ch] = f"corr={float(np.mean(r)):.2f}"
        # Resultado
        if not bad:
            self.bad_detect_status.setText("Nenhum canal ruim detectado.")
            self.bad_detect_status.setStyleSheet(
                f"color: {SIGNAL_TYPE_COLORS['EEG']}; font-weight: bold;")
            self._log("Detecção de canais ruins: OK (nenhum canal flagged)")
            return
        # Marca como inativos
        msg = ", ".join(f"CH{c+1} ({reasons[c]})" for c in bad)
        confirm = QtWidgets.QMessageBox.question(
            self, "Canais ruins detectados",
            f"<b>{len(bad)} canais marcados como ruins:</b><br>{msg}<br><br>"
            "Deseja desabilitá-los automaticamente?",
            QtWidgets.QMessageBox.StandardButton.Yes |
            QtWidgets.QMessageBox.StandardButton.No)
        if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
            for ch in bad:
                if hasattr(self, "channel_checks") and ch < len(self.channel_checks):
                    self.channel_checks[ch].setChecked(False)
            self.bad_detect_status.setText(
                f"{len(bad)} canais desabilitados: {[c+1 for c in bad]}")
            self.bad_detect_status.setStyleSheet(
                f"color: {COLORS['warning']}; font-weight: bold;")
            self._audit_event("bad_channels_detected", channels=[c+1 for c in bad])
            self._log(f"Canais ruins desabilitados: {[c+1 for c in bad]}")

    def _on_reref_changed(self, *_):
        """Aplica o esquema de re-referenciação ao FilterChain."""
        if not hasattr(self, "reref_combo"): return
        mode = self.reref_combo.currentData()
        # Parse canais 1-based -> 0-based
        ref_chs = []
        try:
            raw = self.reref_channels_edit.text().strip()
            if raw:
                for tok in raw.replace(";", ",").split(","):
                    tok = tok.strip()
                    if tok.isdigit():
                        v = int(tok) - 1
                        if 0 <= v < MAX_CHANNELS:
                            ref_chs.append(v)
        except Exception:
            ref_chs = []
        try:
            self.filters.set_reref(mode, ref_chs)
            self._log(f"Re-referenciação: {mode} (canais ref: {ref_chs or 'auto'})")
            self._audit_event("reref_change", mode=mode, ref_channels=ref_chs)
        except Exception as exc:
            self._log(f"Erro ao aplicar re-referenciação: {exc}", error=True)

    def _on_filter_change(self, *_):
        self.filters.notch_enabled    = self.notch_enable.isChecked()
        self.filters.notch_freq       = float(self.notch_freq.currentText())
        self.filters.bandpass_enabled = self.bp_enable.isChecked()
        self.filters.bp_low           = float(self.bp_low.value())
        self.filters.bp_high          = float(self.bp_high.value())
        try:
            self.filters.rebuild()
            self._log(
                f"Filtros: notch={'ON' if self.filters.notch_enabled else 'OFF'} "
                f"@ {self.filters.notch_freq:.0f}Hz | "
                f"bandpass={'ON' if self.filters.bandpass_enabled else 'OFF'} "
                f"{self.filters.bp_low:.1f}-{self.filters.bp_high:.1f}Hz")
            self._audit_event("filter_change",
                              notch=self.filters.notch_enabled,
                              notch_freq=self.filters.notch_freq,
                              bp=self.filters.bandpass_enabled,
                              bp_low=self.filters.bp_low,
                              bp_high=self.filters.bp_high)
        except Exception as exc:
            self._log(f"ERRO no filtro: {exc}", error=True)

    def _on_channel_toggle(self, ch, active):
        self.channel_active[ch] = active
        self._apply_channel_visibility()

    def _toggle_connection(self):
        if self.serial_thread and self.serial_thread.isRunning():
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        mode_idx = self.mode_combo.currentIndex()
        mode = ["hardware", "simulation", "playback"][mode_idx]
        port, playback_path = None, None
        if mode == "hardware":
            port = self.port_combo.currentData()
            if not port:
                self._log("ERRO: nenhuma porta COM válida selecionada", error=True); return
        elif mode == "playback":
            playback_path = self.playback_path_edit.text().strip()
            if not playback_path:
                self._log("ERRO: selecione um arquivo CSV de playback", error=True); return
            port = "PLAYBACK"
        else:
            port = "SIMULACAO"
        try:
            baud = int(self.baud_combo.currentText().strip())
        except ValueError:
            self._log("ERRO: baud rate inválido", error=True); return

        self.buffer.fill(0.0); self.buffer_pos = 0; self.samples_total = 0
        self.accel_buffer.fill(0.0); self.accel_pos = 0
        self.spec_buffer.fill(-80.0); self.spec_pos = 0
        self.filters.reset_state()
        self._clear_marker_lines()
        # Reset timing audit
        self._last_sample_t = None
        self._dt_window.clear()
        self._dt_total = 0
        self._dt_mean = self._dt_jitter = 0.0
        self._dropped_count = 0

        # Passa explicitamente num_channels: a thread precisa saber se devemos
        # operar em 8/16 (Cyton) ou 24/32/.../64 (placa customizada).
        self.serial_thread = SerialReaderThread(
            port=port, baud_rate=baud, mode=mode, playback_path=playback_path,
            daisy=self.daisy_enabled, num_channels=self.num_channels)
        self.serial_thread.data_received.connect(self._on_sample)
        self.serial_thread.error.connect(self._on_thread_error)
        self.serial_thread.connection_state.connect(self._on_connection_state)
        self.serial_thread.progress.connect(self._on_playback_progress)
        self.serial_thread.expansion_detected.connect(self._on_expansion_detected)
        self.serial_thread.start()
        self.connect_btn.setText("■  Desconectar")
        self._log(f"Conectando a {port} @ {baud} (modo={mode}, daisy={self.daisy_enabled})")
        self._audit_event("connect", port=str(port), baud=baud, mode=mode,
                          expansion=self.daisy_enabled, playback=playback_path)
        # Simulação / Playback não têm nada para configurar: se o usuário está
        # na aba Conexão, leva-o direto para Tempo Real para ver o sinal.
        if mode in ("simulation", "playback") and hasattr(self, "_main_tabs"):
            if self._main_tabs.currentIndex() == 0:   # ainda em "Configurar"
                self._goto_realtime_view()

    def _on_expansion_detected(self, n):
        """Disparado pela thread quando detecta canais (ex: CSV com 16ch)."""
        if n != self.num_channels:
            self._log(f"Expansão detectada automaticamente: {n} canais")
            self._set_num_channels(n)

    def _disconnect(self):
        if self.is_recording:
            self._stop_recording()
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread = None
        self.connect_btn.setText("▶  Conectar")
        self._log("Desconectado")
        self._audit_event("disconnect",
                          dt_mean_ms=round(self._dt_mean, 3),
                          dt_jitter_ms=round(self._dt_jitter, 3),
                          dropped=self._dropped_count,
                          samples_total=self.samples_total)

    def _on_connection_state(self, connected):
        if connected:
            self.session_start = time.time()
            self.status_dot.setStyleSheet(f"color: {COLORS['accent']}; font-size: 22pt;")
            self.status_label.setText("CONECTADO")
            self.status_label.setStyleSheet(
                f"color: {COLORS['accent']}; font-weight: bold; "
                f"padding: 0 14px; font-size: 11pt;")
            self.record_btn.setEnabled(True)
            self._log("Conexão estabelecida — recebendo dados")
        else:
            self.status_dot.setStyleSheet(f"color: {COLORS['error']}; font-size: 22pt;")
            self.status_label.setText("DESCONECTADO")
            self.status_label.setStyleSheet(
                f"color: {COLORS['error']}; font-weight: bold; "
                f"padding: 0 14px; font-size: 11pt;")
            self.record_btn.setEnabled(False)
            self.connect_btn.setText("▶  Conectar")
        # Atualiza status bar e título da janela
        self._update_status_state()
        self._update_window_title()

    def _on_thread_error(self, msg):
        # Erro técnico para o log, mensagem amigável para o usuário
        self._log(f"ERRO: {msg}", error=True)
        self._update_status_state("Erro na conexão — veja o log para detalhes")
        # Mensagens amigáveis para erros comuns
        friendly = self._friendly_error_message(msg)
        if friendly:
            QtWidgets.QMessageBox.warning(self, "Problema na conexão", friendly)

    @staticmethod
    def _friendly_error_message(raw_msg):
        """Converte mensagens técnicas em explicações amigáveis."""
        low = raw_msg.lower()
        if "could not open port" in low or "permissionerror" in low:
            return ("Não foi possível abrir a porta COM selecionada.\n\n"
                    "Verifique se:\n"
                    " • O dispositivo está conectado e ligado\n"
                    " • Nenhum outro programa (OpenBCI GUI, Arduino IDE, etc.) "
                    "está usando a porta\n"
                    " • A porta selecionada é a correta")
        if "filenotfounderror" in low or "no such file" in low:
            return ("Arquivo não encontrado.\n\nVerifique se o caminho está "
                    "correto e se o arquivo ainda existe.")
        if "no module" in low:
            return ("Uma biblioteca necessária não está instalada.\n\n"
                    "Detalhe técnico para suporte: " + raw_msg)
        if "csv de playback" in low:
            return ("Arquivo CSV de playback não foi encontrado.\n"
                    "Verifique se selecionou um arquivo válido.")
        return None

    def _notify_error(self, code, detail="", exc=None, blocking=None):
        """Sinaliza um erro catalogado (Erro E0XX) ao usuário: SEMPRE loga e
        reflete na barra de status; abre diálogo modal se for bloqueante."""
        title, _msg, blk = error_info(code)
        if blocking is None:
            blocking = blk
        try:
            if hasattr(self, "status_state_lbl"):
                self.status_state_lbl.setText(f"⚠ Erro {code}: {title}")
            self.statusBar().showMessage(f"Erro {code}: {title}", 8000)
        except Exception:
            pass
        if blocking:
            notify_error(code, detail, parent=self, exc=exc, blocking=True)
        else:
            logging.getLogger("eeg").warning(
                "[%s] %s%s", code, title, (" | " + str(detail)) if detail else "")

    def _on_playback_progress(self, frac):
        self.playback_progress.setValue(int(frac * 1000))

    # ==================================================================
    # Sample handler
    # ==================================================================
    def _on_sample(self, sample, accel):
        # --- Sample timing audit (jitter / drops) ---
        now = time.time()
        if self._last_sample_t is not None:
            dt_ms = (now - self._last_sample_t) * 1000.0
            self._dt_window.append(dt_ms)
            self._dt_total += 1
            expected_ms = 1000.0 / SAMPLE_RATE
            # > 2x esperado = drop detectado
            if dt_ms > 2.0 * expected_ms:
                self._dropped_count += int(round(dt_ms / expected_ms)) - 1
            # média e jitter (sd) da janela
            if len(self._dt_window) >= 5:
                arr = np.fromiter(self._dt_window, dtype=np.float64,
                                  count=len(self._dt_window))
                self._dt_mean   = float(np.mean(arr))
                self._dt_jitter = float(np.std(arr))
        self._last_sample_t = now

        try:
            sample = self.filters.apply_sample(sample)
        except Exception as exc:
            # NAO engolir em silencio: do contrario gravariamos a amostra CRUA
            # rotulada como filtrada. Conta, marca a sessao como degradada e
            # loga com throttle (1a falha + a cada 250).
            self._filter_fail_count = getattr(self, "_filter_fail_count", 0) + 1
            self._filter_degraded = True
            if self._filter_fail_count == 1 or self._filter_fail_count % 250 == 0:
                logging.getLogger("eeg").warning(
                    "Falha na filtragem em tempo real (#%d): %s — amostra gravada CRUA",
                    self._filter_fail_count, exc)
        self.last_accel = accel
        n = len(sample)
        # Buffer EEG (preenche só os canais ativos)
        self.buffer[:n, self.buffer_pos] = sample
        if n < MAX_CHANNELS:
            self.buffer[n:, self.buffer_pos] = 0.0
        self.buffer_pos = (self.buffer_pos + 1) % BUFFER_SIZE
        self.samples_total += 1
        # Acelerômetro
        self.accel_buffer[:, self.accel_pos] = accel
        self.accel_pos = (self.accel_pos + 1) % ACCEL_BUFFER_SIZE
        # CSV
        if self.is_recording and self.csv_writer is not None:
            try:
                ts = time.time() - self.session_start
                marker = self.pending_marker or ""
                self.pending_marker = None
                row = [f"{ts:.4f}"] + [f"{v:.4f}" for v in sample]
                row += [f"{accel[0]:.4f}", f"{accel[1]:.4f}", f"{accel[2]:.4f}", marker]
                self.csv_writer.writerow(row)
            except Exception as exc:
                self._log(f"Erro na gravação CSV: {exc}", error=True)
        if self.udp.enabled:
            t = time.time() - (self.session_start or time.time())
            self.udp.send_sample(t, sample)
        # LSL — só os canais ativos
        if self.lsl.enabled:
            self.lsl.send_sample(sample[:self.num_channels])

    def _ordered_buffer(self):
        if self.samples_total == 0:
            return np.zeros((MAX_CHANNELS, 0))
        if self.samples_total < BUFFER_SIZE:
            return self.buffer[:, : self.samples_total]
        return np.concatenate(
            (self.buffer[:, self.buffer_pos:], self.buffer[:, : self.buffer_pos]),
            axis=1)

    def _ordered_accel(self):
        if self.samples_total == 0:
            return np.zeros((3, 0))
        n = min(self.samples_total, ACCEL_BUFFER_SIZE)
        if self.samples_total < ACCEL_BUFFER_SIZE:
            return self.accel_buffer[:, :n]
        return np.concatenate(
            (self.accel_buffer[:, self.accel_pos:], self.accel_buffer[:, : self.accel_pos]),
            axis=1)

    # ==================================================================
    # Update timers
    # ==================================================================
    def _update_plots(self):
        data = self._ordered_buffer()
        n_samples = data.shape[1]
        if n_samples == 0: return
        t_axis = np.arange(n_samples) / SAMPLE_RATE

        view_idx = self.rt_view_stack.currentIndex() if hasattr(self, "rt_view_stack") else 1

        # Histórico sempre atualiza (aba Histórico) + Individual se ativo.
        # Protegido: uma falha num canal não pode derrubar o timer de 50ms
        # (no PySide6 uma exceção não tratada num slot de timer pode abortar o app).
        try:
            for ch in range(MAX_CHANNELS):
                active = (ch < self.num_channels and self.channel_active[ch])
                if not active:
                    self.channel_curves[ch].setData([], [])
                    self.history_curves[ch].setData([], [])
                    continue
                self.history_curves[ch].setData(t_axis, data[ch])
                if view_idx == 1:
                    self.channel_curves[ch].setData(t_axis, data[ch])

            # Modo Empilhado (montagem clínica) — todos os canais num único plot
            if view_idx == 0 and hasattr(self, "montage_plot"):
                self._update_montage(data)
        except Exception as exc:
            if not getattr(self, "_err_logged_core_plot", False):
                self._log(f"Erro ao atualizar gráficos: {exc}", error=True)
                self._err_logged_core_plot = True

        # Atualizações das modalidades Bio + saídas BCI — sempre protegidas
        # por try/except para que falhas nelas não derrubem o timer principal.
        def _safe(fn_name):
            fn = getattr(self, fn_name, None)
            if fn is None: return
            try:
                fn()
            except Exception as exc:
                flag = f"_err_logged_{fn_name}"
                if not getattr(self, flag, False):
                    self._log(f"Erro {fn_name}: {exc}", error=True)
                    setattr(self, flag, True)

        # EMG view (aba EMG / Músculos)
        if hasattr(self, "emg_rows"):
            _safe("_update_emg_view")
        # ECG view (aba ECG / Coração)
        if hasattr(self, "ecg_raw_plot"):
            _safe("_update_ecg_view")
        # EoG view (aba EoG / Olhos)
        if hasattr(self, "eog_plot"):
            _safe("_update_eog_view")
        # Focus / SSVEP (aba em Analisar)
        if hasattr(self, "focus_fft_plot"):
            _safe("_update_focus_view")
        # EMG Joystick (aba em Analisar)
        if hasattr(self, "_joy_axes"):
            _safe("_update_emg_joystick_view")
        # Acelerômetro · Movimento (sub-aba em Bio)
        if hasattr(self, "accel_history_plot"):
            _safe("_update_accelerometer_view")

        accel_data = self._ordered_accel()
        if accel_data.shape[1] > 0:
            t_a = np.arange(accel_data.shape[1]) / SAMPLE_RATE
            for i in range(3):
                self.accel_curves[i].setData(t_a, accel_data[i])
        self.samples_label.setText(f"Amostras: {self.samples_total:,}".replace(",", "."))
        a = self.last_accel
        self.accel_label.setText(f"g: {a[0]:+.2f}  {a[1]:+.2f}  {a[2]:+.2f}")
        # Timing / jitter / drops
        if hasattr(self, "timing_label"):
            expected = 1000.0 / SAMPLE_RATE
            if self._dt_total >= 5:
                drift_pct = (self._dt_mean - expected) / expected * 100.0
                txt = (f"Δt: {self._dt_mean:5.2f} ms  ±{self._dt_jitter:.2f}  "
                       f"(drift {drift_pct:+.1f}%)")
                if self._dropped_count > 0:
                    txt += f"  drops: {self._dropped_count}"
                # cor: verde se ok, amarelo se jitter alto, vermelho se drops
                if self._dropped_count > 0:
                    col = COLORS["error"]
                elif self._dt_jitter > expected * 0.20:
                    col = COLORS["warning"]
                else:
                    col = COLORS["accent"]
                self.timing_label.setStyleSheet(
                    f"color: {col}; padding: 0 10px; font-family: {FONT_DATA_STACK};")
                self.timing_label.setText(txt)

    def _update_montage(self, data):
        """Desenha todos os canais visíveis empilhados num único plot,
        cada um com offset vertical (estilo montagem clínica EEG)."""
        n_total = data.shape[1]
        win_s = self.rt_window_spin.value()
        win_n = min(n_total, int(win_s * SAMPLE_RATE))
        seg = data[:, -win_n:]
        t = np.arange(win_n) / SAMPLE_RATE
        scale = max(1e-6, self.rt_scale_spin.value())

        vis = [ch for ch in range(MAX_CHANNELS)
               if ch < self.num_channels and self.channel_active[ch]]
        n_vis = len(vis)
        for k, ch in enumerate(vis):
            baseline = (n_vis - 1 - k)
            # Centraliza no baseline e escala: 1 "slot" = `scale` µV
            y = (seg[ch] - float(np.mean(seg[ch]))) / scale + baseline
            self.montage_curves[ch].setData(t, y)
        # Esconde curvas de canais não visíveis
        for ch in range(MAX_CHANNELS):
            if ch not in vis:
                self.montage_curves[ch].setData([], [])
        self.montage_plot.setXRange(0, win_s, padding=0)
        # Atualiza ticks só se mudou o nº de canais visíveis
        if getattr(self, "_montage_last_nvis", -1) != n_vis:
            self._montage_last_nvis = n_vis
            self._refresh_montage_layout()

    def _update_analysis(self):
        data = self._ordered_buffer()
        if data.shape[1] < SAMPLE_RATE: return
        ch = self.analysis_channel.currentIndex()
        if ch >= self.num_channels: ch = 0
        freqs, fft_vals = SignalProcessor.compute_fft(data[ch])
        if freqs.size:
            self.fft_curve.setData(freqs, fft_vals)
        powers = SignalProcessor.compute_band_powers(data[ch])
        self.band_bars.setOpts(height=list(powers.values()))
        for c in range(MAX_CHANNELS):
            if c >= self.num_channels:
                self.stats_table.item(c, 1).setText("--")
                self.stats_table.item(c, 2).setText("--")
                self.stats_table.item(c, 3).setText("--")
                continue
            stats = SignalProcessor.compute_statistics(data[c])
            self.stats_table.item(c, 1).setText(f"{stats['mean']:+.3f}")
            self.stats_table.item(c, 2).setText(f"{stats['std']:.3f}")
            self.stats_table.item(c, 3).setText(f"{stats['rms']:.3f}")

    def _update_spectrogram(self):
        data = self._ordered_buffer()
        if data.shape[1] < SAMPLE_RATE: return
        for ch in range(self.num_channels):
            col = SignalProcessor.compute_psd_column(data[ch])
            self.spec_buffer[ch, :, self.spec_pos] = col
        self.spec_pos = (self.spec_pos + 1) % SPEC_FRAMES
        ch = self.spec_channel.currentIndex()
        if ch >= self.num_channels: ch = 0
        view = np.concatenate(
            (self.spec_buffer[ch, :, self.spec_pos:],
             self.spec_buffer[ch, :, : self.spec_pos]),
            axis=1)
        self.spec_image.setImage(view.T, autoLevels=False)
        self.spec_image.setRect(QtCore.QRectF(0.0, 0.0,
                                              SPEC_FRAMES * 0.25, SPEC_FMAX))

    def _update_topology(self):
        data = self._ordered_buffer()
        if data.shape[1] < SAMPLE_RATE: return
        band_name = self.topo_band_combo.currentText()
        low, high = EEG_BANDS[band_name]
        powers = []
        for c in range(MAX_CHANNELS):
            if c < self.num_channels:
                powers.append(SignalProcessor.compute_band_power(data[c], low, high))
            else:
                powers.append(0.0)
        self.head_plot.set_powers(powers, band_name)
        ch = self.topo_channel_combo.currentIndex()
        if ch >= self.num_channels: ch = 0
        try:
            focus = SignalProcessor.compute_focus_index(data[ch])
            self.focus_meter.update_value(focus)
        except Exception: pass
        emg_ch = self.emg_widget.channel_combo.currentIndex()
        if emg_ch >= self.num_channels: emg_ch = 0
        n_last = min(int(SAMPLE_RATE * 3), data.shape[1])
        self.emg_widget.update_signal(data[emg_ch, -n_last:])

    # ==================================================================
    # Qualidade de sinal por canal — LEDs no header
    # ==================================================================
    def _update_channel_quality(self):
        """Classifica cada canal como OK / NOISY / BAD com base em:
        - amplitude saturada (clipping)
        - variância fora da faixa de EEG (~0.5–200 µV²)
        - razão de potência em ~50/60 Hz (rede elétrica) vs total
        """
        data = self._ordered_buffer()
        n = data.shape[1]
        if n < SAMPLE_RATE // 2:
            return
        last = data[:, -int(SAMPLE_RATE * 2):]  # últimos 2s
        # Faixas absolutas — Cyton: signal aceitável típico é ~10–80 µV RMS
        # Saturação: >2000 µV pico-a-pico = ruim
        # Cores semânticas (escurecidas p/ contraste tanto no claro quanto no
        # escuro) e SÍMBOLO por estado (acessibilidade — não depender só de cor).
        n_ok = n_noisy = n_bad = 0
        for ch in range(MAX_CHANNELS):
            if ch >= self.num_channels:
                self.quality_leds[ch].setVisible(False)
                continue
            self.quality_leds[ch].setVisible(True)
            try:
                x = last[ch]
                pp = float(np.max(x) - np.min(x))
                rms = float(np.sqrt(np.mean(x * x)))
                # Potência relativa em 55–65 Hz vs total (line noise)
                if len(x) >= SAMPLE_RATE:
                    f, psd = scipy_signal.welch(x, fs=SAMPLE_RATE,
                                                 nperseg=min(256, len(x)))
                    line_mask  = (f >= 55) & (f <= 65)
                    total_mask = (f >= 1)  & (f <= 80)
                    line_pwr  = float(np.sum(psd[line_mask]))
                    total_pwr = float(np.sum(psd[total_mask]))
                    line_ratio = line_pwr / max(total_pwr, 1e-12)
                else:
                    line_ratio = 0.0
                # Classificação heurística -> (cor, símbolo, severidade, texto)
                if pp > 2000 or rms > 500:
                    color, glyph, sev, status = ("#d4364f", "■", "bad",
                        f"SATURADO (pp={pp:.0f}µV)")
                elif rms < 0.5:
                    color, glyph, sev, status = ("#d4364f", "■", "bad",
                        f"FLAT (rms={rms:.2f}µV)")
                elif line_ratio > 0.40 or rms > 150:
                    color, glyph, sev, status = ("#d68a00", "▲", "noisy",
                        f"RUIDOSO (rms={rms:.0f}µV, rede {line_ratio*100:.0f}%)")
                else:
                    color, glyph, sev, status = ("#1d9e75", "●", "ok",
                        f"OK (rms={rms:.0f}µV)")
                if sev == "bad":      n_bad += 1
                elif sev == "noisy":  n_noisy += 1
                else:                 n_ok += 1
                ele = self.config.channel_mapping[ch] if ch < len(self.config.channel_mapping) else f"CH{ch+1}"
                self.quality_leds[ch].setText(glyph)
                self.quality_leds[ch].setStyleSheet(f"color: {color}; font-size: 11pt;")
                self.quality_leds[ch].setToolTip(f"CH{ch+1} ({ele}): {status}")
            except Exception:
                pass

        # ---- Semáforo agregado "Pronto para gravar" (cor + símbolo + rótulo) ----
        if hasattr(self, "quality_summary"):
            active = n_ok + n_noisy + n_bad
            if active == 0:
                txt, col, tip = "—", "#888", "Sinal: aguardando dados"
            elif n_bad:
                txt, col = f"■ {n_bad} ruim", "#d4364f"
                tip = f"Sinal: {n_bad} canal(is) ruim/saturado — verifique eletrodos"
            elif n_noisy:
                txt, col = f"▲ {n_noisy} ruidoso{'s' if n_noisy > 1 else ''}", "#d68a00"
                tip = f"Sinal: {n_noisy} canal(is) ruidoso(s) — atenção"
            else:
                txt, col, tip = "● Sinal OK", "#1d9e75", "Sinal pronto para gravar"
            self.quality_summary.setText(txt)
            self.quality_summary.setToolTip(tip)
            self.quality_summary.setStyleSheet(
                f"color: {col}; font-size: 9pt; font-weight: bold; padding: 0 8px;")
        # Re-flui o header responsivo: a telemetria muda de largura ao longo da
        # sessão (Amostras, g, drift) sem disparar resize — reavaliamos aqui.
        try:
            self._update_header_responsive()
        except Exception:
            pass

    # ==================================================================
    # Markers
    # ==================================================================
    def _clear_marker_lines(self):
        if hasattr(self, "channel_marker_lines"):
            for ch_lines in self.channel_marker_lines:
                for line in ch_lines:
                    try:
                        for plot in self.channel_plots:
                            plot.removeItem(line)
                    except Exception: pass
                ch_lines.clear()
        # Limpa markers do modo Empilhado
        if hasattr(self, "montage_marker_lines"):
            for _t, mline in self.montage_marker_lines:
                try: self.montage_plot.removeItem(mline)
                except Exception: pass
            self.montage_marker_lines.clear()

    def _inject_marker_from_edit(self):
        label = self.marker_edit.text().strip()
        if not label:
            self._log("Marker vazio ignorado", error=True); return
        self._inject_marker_text(label)
        self.marker_edit.clear()

    def _inject_marker_text(self, label):
        t = (time.time() - self.session_start) if self.session_start else 0.0
        self.markers.append((t, label))
        self.pending_marker = label
        self._audit_event("marker", label=label, t_session=round(t, 3))
        # events.csv — mapeia o marker à amostra atual (linha do data.csv)
        if self.is_recording:
            self.events_logger.log(label, self.samples_total, t)
        self.markers_view.append(
            f'<span style="color:{COLORS["text_dim"]}">[{t:7.2f}s]</span> '
            f'<span style="color:{COLORS["accent"]}; font-weight:bold;">{label}</span>')
        if self.udp.enabled:
            self.udp.send_marker(t, label)
        if self.lsl.enabled:
            self.lsl.send_marker(label)
        self._log(f"Marker: {label} @ {t:.2f}s")
        if hasattr(self, "channel_plots"):
            for ch, plot in enumerate(self.channel_plots):
                line = pg.InfiniteLine(
                    pos=(self.samples_total / SAMPLE_RATE), angle=90,
                    pen=pg.mkPen(COLORS["warning"], width=1, style=QtCore.Qt.PenStyle.DashLine))
                plot.addItem(line)
                self.channel_marker_lines[ch].append(line)
        # Linha de marker no modo Empilhado (na posição = fim da janela)
        if hasattr(self, "montage_plot"):
            win_s = self.rt_window_spin.value() if hasattr(self, "rt_window_spin") else 5
            mline = pg.InfiniteLine(
                pos=win_s, angle=90,
                pen=pg.mkPen(COLORS["warning"], width=1.4,
                              style=QtCore.Qt.PenStyle.DashLine),
                label=label,
                labelOpts={"color": COLORS["warning"], "position": 0.95})
            self.montage_plot.addItem(mline)
            self.montage_marker_lines.append((time.time(), mline))
            # Mantém só os últimos 12 markers visíveis (limpa antigos)
            while len(self.montage_marker_lines) > 12:
                _t, old = self.montage_marker_lines.pop(0)
                try: self.montage_plot.removeItem(old)
                except Exception: pass

    # ==================================================================
    # Gravação
    # ==================================================================
    def _toggle_recording(self):
        if self.is_recording: self._stop_recording()
        else: self._start_recording()

    def _start_recording(self):
        # ---- Pasta de sessão customizada ----
        save_root = self.config.save_directory
        try:
            os.makedirs(save_root, exist_ok=True)
        except Exception as exc:
            self._log(f"Diretório de salvamento inválido ({save_root}): {exc} — "
                      f"caindo para o padrão", error=True)
            save_root = DEFAULT_SAVE_DIRECTORY
            os.makedirs(save_root, exist_ok=True)
            self.config.save_directory = save_root
            self.config.save()
        self.session_name = self._build_session_name()
        # Se houver voluntário ativo, a sessão vai para volunteers/<VID_Nome>/
        vol = self.volunteers.current() if hasattr(self, "volunteers") else None
        if vol:
            self.volunteers.set_base_dir(save_root)
            vol_dir = self.volunteers.current_dir()
            os.makedirs(vol_dir, exist_ok=True)
            # prefixa o nome da sessão com o VID para rastreabilidade
            self.session_name = f"{vol.get('vid','V00')}_{self.session_name}"
            self.current_session_dir = os.path.join(vol_dir, self.session_name)
        else:
            self.current_session_dir = os.path.join(save_root, self.session_name)
        # garante unicidade se já existir
        suffix = 1
        base = self.current_session_dir
        while os.path.exists(self.current_session_dir):
            self.current_session_dir = f"{base}_{suffix}"
            suffix += 1
        os.makedirs(self.current_session_dir, exist_ok=True)
        os.makedirs(os.path.join(self.current_session_dir, "snapshots"), exist_ok=True)

        csv_path    = os.path.join(self.current_session_dir, "data.csv")
        log_path    = os.path.join(self.current_session_dir, "session.log.txt")
        meta_path   = os.path.join(self.current_session_dir, "summary.json")
        events_path = os.path.join(self.current_session_dir, "events.csv")

        # events.csv — mapeia markers → linhas do data.csv (análise futura)
        self.events_logger.start(events_path)

        # Snapshot da ficha do voluntário dentro da sessão
        if vol:
            try:
                with open(os.path.join(self.current_session_dir,
                                       "volunteer_profile.json"),
                          "w", encoding="utf-8") as f:
                    json.dump(vol, f, ensure_ascii=False, indent=2)
            except Exception: pass
        try:
            self.csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            # Cabeçalho usa os nomes do mapeamento (ex: Fp1_uV) em vez de CHn_uV
            mapping = self.config.channel_mapping
            ch_names = [f"{mapping[i]}_uV" if i < len(mapping) else f"CH{i+1}_uV"
                        for i in range(self.num_channels)]
            header = (["timestamp_s"] + ch_names
                      + ["ax_g", "ay_g", "az_g", "marker"])
            self.csv_writer.writerow(header)
            self.log_file = open(log_path, "w", encoding="utf-8")
            self.log_file.write(
                f"=== Sessão '{self.session_name}' iniciada em {datetime.now().isoformat()} ===\n"
                f"Gerado por: {APP_NAME} v{APP_VERSION} ({APP_EDITION}) — {CODE_URL}\n"
                f"Sujeito: {self.config.subject or '(não informado)'}\n"
                f"Porta/modo: {self.port_combo.currentText()} | {self.mode_combo.currentText()}\n"
                f"Baud: {self.baud_combo.currentText()}\n"
                f"Número de canais: {self.num_channels} "
                f"({'com módulo de expansão' if self.num_channels > BASE_CHANNELS else 'placa base'})\n"
                f"Sample Rate: {SAMPLE_RATE} Hz\n"
                f"Mapeamento: {mapping[:self.num_channels]}\n"
                f"Canais ativos: "
                f"{[mapping[i] for i, a in enumerate(self.channel_active[:self.num_channels]) if a]}\n"
                f"Filtros: notch={'ON' if self.filters.notch_enabled else 'OFF'} "
                f"@ {self.filters.notch_freq:.0f}Hz | "
                f"bandpass={'ON' if self.filters.bandpass_enabled else 'OFF'} "
                f"{self.filters.bp_low:.1f}-{self.filters.bp_high:.1f}Hz\n"
                f"Tema: {self.config.theme}\n"
                f"Snapshot interval: {self.config.snapshot_interval}s\n"
                f"Pasta: {self.current_session_dir}\n\n")
            self.log_file.flush()
            # summary.json
            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump({
                        # Proveniencia (reprodutibilidade/auditoria): qual versao
                        # do app e de onde veio o codigo geraram este registro.
                        "generated_by": {
                            "app":          APP_NAME,
                            "app_version":  APP_VERSION,
                            "app_edition":  APP_EDITION,
                            "code_url":     CODE_URL,
                        },
                        "session_name": self.session_name,
                        "started_at":   datetime.now().isoformat(),
                        "subject":      self.config.subject,
                        "num_channels": self.num_channels,
                        "mapping":      mapping[:self.num_channels],
                        "sample_rate":  SAMPLE_RATE,
                        "filters": {
                            "notch_on":   self.filters.notch_enabled,
                            "notch_freq": self.filters.notch_freq,
                            "bp_on":      self.filters.bandpass_enabled,
                            "bp_low":     self.filters.bp_low,
                            "bp_high":    self.filters.bp_high,
                        },
                        "theme":        self.config.theme,
                        "volunteer":    (self.volunteers.current()
                                         if hasattr(self, "volunteers") else None),
                    }, f, ensure_ascii=False, indent=2)
            except Exception: pass
        except OSError as exc:
            self._log(f"Falha ao criar arquivo: {exc}", error=True)
            self._close_record_files(); return

        self.is_recording = True
        self._snapshot_counter = 0
        self.record_btn.setText("■  Parar Gravação")
        self.record_btn.setStyleSheet(
            f"background-color: {COLORS['error']}; color: white; "
            f"border-color: {COLORS['error']};")
        self.rec_indicator.setText("● REC")
        self._log(f"Gravação iniciada — {self.session_name}/data.csv "
                  f"({self.num_channels} canais)")
        self._update_status_state()
        self._update_window_title()
        # Reabre o audit log na pasta da nova sessão
        self._close_audit_log()
        self._open_audit_log()
        self._audit_event("recording_start",
                          path=csv_path, num_channels=self.num_channels,
                          mapping=self.config.channel_mapping[:self.num_channels],
                          filters={"notch": self.filters.notch_enabled,
                                   "notch_freq": self.filters.notch_freq,
                                   "bp": self.filters.bandpass_enabled,
                                   "bp_low": self.filters.bp_low,
                                   "bp_high": self.filters.bp_high})

        # Snapshot inicial + timer para snapshots periódicos
        if self.config.snapshot_interval > 0:
            self._save_snapshot(reason="início")
            if not hasattr(self, "snapshot_timer") or self.snapshot_timer is None:
                self.snapshot_timer = QTimer(self)
                self.snapshot_timer.timeout.connect(
                    lambda: self._save_snapshot(reason="auto")
                )
            self.snapshot_timer.start(self.config.snapshot_interval * 1000)

    def _stop_recording(self):
        # Snapshot final antes de fechar (capta o estado atual)
        try:
            if self.current_session_dir:
                self._save_snapshot(reason="fim")
        except Exception: pass
        if hasattr(self, "snapshot_timer") and self.snapshot_timer is not None:
            self.snapshot_timer.stop()
        self.is_recording = False
        if self.log_file:
            try:
                self.log_file.write(
                    f"\n=== Sessão finalizada em {datetime.now().isoformat()} ===\n"
                    f"Total de amostras gravadas: {self.samples_total}\n"
                    f"Total de marcadores: {len(self.markers)}\n"
                    f"Snapshots capturados: {self._snapshot_counter}\n")
            except Exception: pass
        self._close_record_files()
        self.record_btn.setText("●  Iniciar Gravação")
        self.record_btn.setStyleSheet("")
        self.rec_indicator.setText("")
        self._log(f"Gravação finalizada — pasta: {self.session_name}")
        self._update_status_state()
        self._update_window_title()
        self._audit_event("recording_stop",
                          samples=self.samples_total,
                          markers=len(self.markers),
                          dropped=self._dropped_count,
                          dt_mean_ms=round(self._dt_mean, 3),
                          dt_jitter_ms=round(self._dt_jitter, 3))
        # Fecha events.csv
        self.events_logger.stop()
        # Registra execução no progress.json do voluntário ativo
        if hasattr(self, "volunteers") and self.volunteers.current():
            try:
                self.volunteers.add_execution(
                    self.session_name, self.samples_total, len(self.markers))
                if hasattr(self, "_volunteer_show_history"):
                    self._volunteer_show_history()
            except Exception as exc:
                self._log(f"Falha atualizando progresso do voluntário: {exc}",
                          error=True)
        # SHA-256 + nonce (prova de integridade — validação clínica)
        self._sign_session_files()

    def _sign_session_files(self):
        """Calcula SHA-256 do data.csv + nonce + atualiza summary.json.
        Permite provar mais tarde que o arquivo não foi alterado."""
        if not self.current_session_dir: return
        csv_path  = os.path.join(self.current_session_dir, "data.csv")
        meta_path = os.path.join(self.current_session_dir, "summary.json")
        if not os.path.exists(csv_path): return
        try:
            h = hashlib.sha256()
            with open(csv_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            digest = h.hexdigest()
            nonce  = secrets.token_hex(16)
            n_bytes = os.path.getsize(csv_path)
            integrity = {
                "data_csv_sha256": digest,
                "data_csv_bytes":  n_bytes,
                "nonce":           nonce,
                "signed_at":       datetime.now().isoformat(timespec="milliseconds"),
            }
            # Mescla com summary.json
            meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception: pass
            meta["integrity"] = integrity
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            # Tambem grava em arquivo proprio para conveniência
            sig_path = os.path.join(self.current_session_dir, "data.csv.sha256")
            with open(sig_path, "w", encoding="utf-8") as f:
                f.write(f"{digest}  data.csv\n")
                f.write(f"# nonce: {nonce}\n")
                f.write(f"# bytes: {n_bytes}\n")
                f.write(f"# signed_at: {integrity['signed_at']}\n")
            self._log(f"✓ Integridade: SHA-256 {digest[:16]}... salvo em summary.json")
            self._audit_event("session_signed",
                              sha256=digest, nonce=nonce, bytes=n_bytes)
        except Exception as exc:
            self._log(f"Falha ao assinar sessão: {exc}", error=True)

    # ==================================================================
    # Snapshot — salva PNGs do estado atual de cada visualização
    # ==================================================================
    def _save_snapshot(self, reason="manual"):
        """Salva imagens das visualizações na pasta da sessão atual."""
        if not self.current_session_dir:
            return
        snap_dir = os.path.join(self.current_session_dir, "snapshots")
        ts = datetime.now().strftime("%H-%M-%S")
        self._snapshot_counter += 1
        idx = self._snapshot_counter

        def _save(widget, name):
            try:
                pm = widget.grab()
                pm.save(os.path.join(snap_dir, f"{idx:04d}_{ts}_{name}.png"), "PNG")
            except Exception as exc:
                pass

        # Captura cada visualização principal
        if hasattr(self, "channels_layout"): _save(self.channels_layout, "tempo_real")
        if hasattr(self, "fft_plot"):        _save(self.fft_plot,        "fft")
        if hasattr(self, "band_plot"):       _save(self.band_plot,       "bandas")
        if hasattr(self, "stats_table"):     _save(self.stats_table,     "estatísticas")
        if hasattr(self, "head_plot"):       _save(self.head_plot,       "head_plot")
        if hasattr(self, "spec_widget"):     _save(self.spec_widget,     "espectrograma")
        if hasattr(self, "history_plot"):    _save(self.history_plot,    "histórico")
        if hasattr(self, "accel_plot"):      _save(self.accel_plot,      "acelerômetro")
        self._log(f"Snapshot #{idx} ({reason}) salvo em snapshots/")

    def _close_record_files(self):
        if self.csv_file:
            try: self.csv_file.close()
            except: pass
        if self.log_file:
            try: self.log_file.close()
            except: pass
        self.csv_file = self.csv_writer = self.log_file = None

    # ==================================================================
    # UDP
    # ==================================================================
    def _toggle_udp(self, checked):
        if checked:
            self.udp.host = self.udp_host_edit.text().strip() or "127.0.0.1"
            self.udp.port = self.udp_port_spin.value()
            ok = self.udp.start()
            if ok:
                self.udp_toggle_btn.setText("Parar streaming")
                self._log(f"UDP streaming ON → {self.udp.host}:{self.udp.port}")
            else:
                self.udp_toggle_btn.setChecked(False)
                self._log("Falha ao iniciar UDP", error=True)
        else:
            self.udp.stop()
            self.udp_toggle_btn.setText("Iniciar streaming")
            self._log("UDP streaming OFF")

    # ==================================================================
    # LSL Receiver — assinar streams externos (markers de PsychoPy/OpenViBE)
    # ==================================================================
    def _lslr_resolve(self):
        """Procura streams LSL na rede e popula combo."""
        if not HAS_LSL:
            return
        try:
            import pylsl  # lazy
            streams = pylsl.resolve_streams(wait_time=2.0)
        except Exception as exc:
            self.lslr_status_lbl.setText(f"Falha: {exc}")
            self.lslr_status_lbl.setStyleSheet(f"color: {COLORS['error']};")
            return
        self._lslr_streams_found = streams
        self.lslr_streams_combo.clear()
        if not streams:
            self.lslr_streams_combo.addItem("(nenhum stream encontrado)", -1)
            self.lslr_status_lbl.setText(
                "Nenhum stream LSL ativo. Verifique se o app emissor "
                "(PsychoPy/OpenViBE/etc.) está rodando.")
            self.lslr_status_lbl.setStyleSheet(f"color: {COLORS['warning']};")
            return
        for i, s in enumerate(streams):
            name = s.name() or "?"
            stype = s.type() or "?"
            srate = s.nominal_srate()
            self.lslr_streams_combo.addItem(
                f"{name} · {stype} · {srate:.0f} Hz", i
            )
        self.lslr_status_lbl.setText(
            f"{len(streams)} stream(s) encontrado(s). "
            "Selecione um e clique 'Assinar'.")
        self.lslr_status_lbl.setStyleSheet(f"color: {SIGNAL_TYPE_COLORS['EEG']};")
        self._log(f"LSL: {len(streams)} stream(s) descoberto(s)")

    def _lslr_toggle_subscribe(self, checked):
        """Assina/cancela assinatura do stream selecionado."""
        if not HAS_LSL:
            return
        if not checked:
            # Cancela assinatura
            if self._lslr_timer:
                self._lslr_timer.stop()
                self._lslr_timer = None
            if self._lslr_inlet:
                try: self._lslr_inlet.close_stream()
                except Exception: pass
                self._lslr_inlet = None
            self.lslr_subscribe_btn.setText("Assinar (markers)")
            self.lslr_status_lbl.setText("Assinatura cancelada.")
            self.lslr_status_lbl.setStyleSheet(f"color: {COLORS['text_dim']};")
            return
        # Assina
        idx = self.lslr_streams_combo.currentData()
        if idx is None or idx < 0 or idx >= len(self._lslr_streams_found):
            self.lslr_subscribe_btn.setChecked(False)
            QtWidgets.QMessageBox.information(
                self, "Sem stream",
                "Selecione um stream da lista primeiro (clique 'Procurar streams LSL').")
            return
        try:
            import pylsl  # lazy
            self._lslr_inlet = pylsl.StreamInlet(self._lslr_streams_found[idx])
            self.lslr_subscribe_btn.setText("Cancelar assinatura")
            self.lslr_status_lbl.setText(
                f"Assinado: {self.lslr_streams_combo.currentText()}. "
                "Markers serão injetados na sessão automaticamente."
            )
            self.lslr_status_lbl.setStyleSheet(
                f"color: {SIGNAL_TYPE_COLORS['EEG']}; font-weight: bold;")
            # Timer de pull
            self._lslr_timer = QTimer(self)
            self._lslr_timer.timeout.connect(self._lslr_pull)
            self._lslr_timer.start(50)  # 20 Hz
            self._audit_event("lsl_receiver_subscribed",
                              stream=self.lslr_streams_combo.currentText())
        except Exception as exc:
            self.lslr_subscribe_btn.setChecked(False)
            self.lslr_status_lbl.setText(f"Falha ao assinar: {exc}")
            self.lslr_status_lbl.setStyleSheet(f"color: {COLORS['error']};")

    def _lslr_pull(self):
        """Busca novas amostras do stream LSL e injeta como markers."""
        if not self._lslr_inlet: return
        try:
            sample, ts = self._lslr_inlet.pull_sample(timeout=0.0)
            if sample is not None:
                # Cada sample pode ter múltiplos canais — concatena como label
                lbl = str(sample[0]) if len(sample) == 1 else "|".join(str(v) for v in sample)
                self.pending_marker = f"LSL:{lbl}"
                self._log(f"LSL marker recebido: {lbl}")
        except Exception:
            pass

    def _toggle_lsl(self, checked):
        if checked:
            self.lsl.stream_name = self.lsl_name_edit.text().strip() or "EEG_Data_Collector"
            ch_labels = self.config.channel_mapping[:self.num_channels]
            ok = self.lsl.start(num_channels=self.num_channels,
                                 sample_rate=SAMPLE_RATE,
                                 ch_labels=ch_labels)
            if ok:
                self.lsl_toggle_btn.setText("Parar LSL")
                self._log(f"LSL ON → stream '{self.lsl.stream_name}' "
                          f"({self.num_channels}ch @ {SAMPLE_RATE} Hz)")
                self._audit_event("lsl_start", stream=self.lsl.stream_name)
            else:
                self.lsl_toggle_btn.setChecked(False)
                self._log("Falha ao iniciar LSL", error=True)
        else:
            self.lsl.stop()
            self.lsl_toggle_btn.setText("Iniciar LSL")
            self._log("LSL OFF")
            self._audit_event("lsl_stop")

    # ==================================================================
    # Hardware
    # ==================================================================
    def _send_hardware_command(self):
        cmd = self.hw_cmd_edit.text().strip()
        if not cmd: return
        self._send_quick_command(cmd)

    def _send_quick_command(self, cmd):
        if self.serial_thread and self.serial_thread.isRunning() and self.serial_thread.mode == "hardware":
            ok = self.serial_thread.send_command(cmd)
            if ok: self._log(f"[HW] Enviado: {cmd!r}")
            else:  self._log(f"[HW] Falha ao enviar: {cmd!r}", error=True)
        else:
            self._log("[HW] Não conectado em modo Hardware — comando ignorado", error=True)

    def _apply_hardware_settings_all(self):
        if not (self.serial_thread and self.serial_thread.isRunning()
                and self.serial_thread.mode == "hardware"):
            self._log("[HW] Conecte em modo Hardware antes de aplicar", error=True); return
        for ch in range(self.num_channels):
            cmd = self._build_channel_command(ch)
            self.serial_thread.send_command(cmd)
            self._log(f"[HW] CH{ch+1} → {cmd}")

    def _build_channel_command(self, ch):
        # Cyton CH1-8 usa digitos "1".."8";
        # Módulo expansão CH9-16 usa codigos "Q"/"W"/"E"/"R"/"T"/"Y"/"U"/"I"
        if ch < 8:
            ch_code = str(ch + 1)
        else:
            ch_code = "QWERTYUI"[ch - 8]
        pwr  = self.hw_power[ch].currentData()
        gain = self.hw_gain[ch].currentData()
        inp  = self.hw_input[ch].currentData()
        bias = self.hw_bias[ch].currentData()
        srb2 = self.hw_srb2[ch].currentData()
        srb1 = self.hw_srb1[ch].currentData()
        return f"x{ch_code}{pwr}{gain}{inp}{bias}{srb2}{srb1}X"

    # ==================================================================
    # Log + close
    # ==================================================================
    def _log(self, msg, error=False):
        ts = datetime.now().strftime("%H:%M:%S")
        color = COLORS["error"] if error else COLORS["accent"]
        self.log_view.append(
            f'<span style="color:{COLORS["text_dim"]}">[{ts}]</span> '
            f'<span style="color:{color}">{msg}</span>')
        if self.log_file:
            try:
                self.log_file.write(f"[{ts}] {msg}\n"); self.log_file.flush()
            except Exception: pass

    # ==================================================================
    # Recursos de usabilidade (barra de status, drag&drop, atalhos, sobre)
    # ==================================================================
    # ==================================================================
    # Visibilidade de abas por modo de aquisição
    # ==================================================================
    # Cada modo (EEG/EMG/ECG/EoG/Hibrido) define quais sub-abas fazem sentido.
    # Ex: Topografia / ERP / ERS-ERD / Conectividade são EEG-only — não tem
    # significado para um sinal EMG (não há referência espacial 10-20).
    MODE_TAB_VISIBILITY = {
        "EEG": {
            "view":    {"Tempo Real", "Topografia", "Espectrograma",
                        "Histórico", "Layout Custom"},
            "analyse": {"Análises", "Offline", "ERP", "Conectividade",
                        "ERS/ERD", "Focus / SSVEP", "BCI Trainer (MI)"},
            "bio":     set(),  # esconde Bio inteiro
        },
        "EMG": {
            "view":    {"Tempo Real", "Bio (EMG/ECG/EoG)",
                        "Histórico", "Layout Custom"},
            "analyse": {"Análises", "Offline", "EMG Joystick"},
            "bio":     {"EMG · Músculos", "Acel · Movimento"},
        },
        "ECG": {
            "view":    {"Tempo Real", "Bio (EMG/ECG/EoG)",
                        "Histórico", "Layout Custom"},
            "analyse": {"Análises", "Offline"},
            "bio":     {"ECG · Coração", "Acel · Movimento"},
        },
        "EoG": {
            "view":    {"Tempo Real", "Bio (EMG/ECG/EoG)",
                        "Histórico", "Layout Custom"},
            "analyse": {"Análises", "Offline"},
            "bio":     {"EoG · Olhos", "Acel · Movimento"},
        },
        # Híbrido — tudo visível
        "Hibrido": None,
    }

    def _apply_signal_mode_visibility(self, acq_type):
        """Mostra/oculta sub-abas conforme o tipo de aquisição.

        Para modos não-EEG, esconde abas que dependem do sistema 10-20
        (Topografia, ERP, ERS/ERD, Conectividade, Focus/SSVEP).
        Para EEG puro, esconde Bio Multimodal e EMG Joystick.
        Para Híbrido, tudo visível.
        """
        if not hasattr(self, "_sub_tabs"):
            return
        rules = self.MODE_TAB_VISIBILITY.get(acq_type)
        if rules is None:
            # Híbrido ou desconhecido — mostra tudo
            for key in ("setup", "view", "analyse", "system"):
                sub = self._sub_tabs.get(key)
                if sub:
                    for i in range(sub.count()):
                        sub.setTabVisible(i, True)
            if hasattr(self, "bio_tabs"):
                for i in range(self.bio_tabs.count()):
                    self.bio_tabs.setTabVisible(i, True)
            return

        # Títulos canônicos (em pt-BR) por índice — espelha _build_ui
        title_map = {
            "view":    ["Tempo Real", "Topografia", "Espectrograma",
                        "Bio (EMG/ECG/EoG)", "Histórico", "Layout Custom"],
            "analyse": ["Análises", "Offline", "ERP", "Conectividade",
                        "ERS/ERD", "Focus / SSVEP", "EMG Joystick",
                        "BCI Trainer (MI)"],
            "bio":     ["EMG · Músculos", "ECG · Coração",
                        "EoG · Olhos", "Acel · Movimento"],
        }

        for key in ("view", "analyse"):
            sub = self._sub_tabs.get(key)
            if not sub: continue
            allowed = rules.get(key, set())
            titles = title_map[key]
            for i, t in enumerate(titles):
                if i >= sub.count(): break
                sub.setTabVisible(i, t in allowed)
            # Seleciona primeira aba visível para o usuário não ficar em uma
            # aba oculta (acontece ao mudar de modo Híbrido para EEG, por ex)
            for i in range(sub.count()):
                if sub.isTabVisible(i):
                    sub.setCurrentIndex(i)
                    break

        # Bio sub-tabs internas
        if hasattr(self, "bio_tabs"):
            allowed_bio = rules.get("bio", set())
            for i, t in enumerate(title_map["bio"]):
                if i >= self.bio_tabs.count(): break
                self.bio_tabs.setTabVisible(i, t in allowed_bio)
            # Seleciona primeira visível
            for i in range(self.bio_tabs.count()):
                if self.bio_tabs.isTabVisible(i):
                    self.bio_tabs.setCurrentIndex(i)
                    break

        self._log(f"Modo de aquisição aplicado: {acq_type} — "
                  f"abas reorganizadas (não-EEG ocultam Topografia/ERP/etc).")

    # ==================================================================
    # Launcher integration — aplica a escolha vinda da LauncherScreen
    # ==================================================================
    def apply_launcher_choice(self, choice):
        """Aplica a escolha do usuário feita na tela inicial.

        choice dict:
            mode             -> 'live' | 'offline' | 'bci' | 'sim' | 'settings'
            port             -> 'COM3' ou None
            num_channels     -> 8..64 (passo de 8) escolhido no Pré-Flight
            expansion_16ch   -> bool (compatibilidade legada)
            acquisition_type -> 'EEG' | 'EMG' | 'ECG' | 'Hibrido'
            volunteer_dir    -> nome da pasta do voluntário ou None
            selected_csv     -> caminho de CSV a abrir no Offline (opcional)
        """
        if not choice:
            return
        self._pending_launcher_choice = choice
        try:
            # 1) Porta COM
            port = choice.get("port")
            if port and hasattr(self, "port_combo"):
                idx = self.port_combo.findData(port)
                if idx >= 0:
                    self.port_combo.setCurrentIndex(idx)

            # 2) Modo de aquisição (Hardware/Simulação/Playback)
            mode = choice.get("mode", "live")
            if hasattr(self, "mode_combo"):
                if mode == "sim":
                    self.mode_combo.setCurrentIndex(1)  # Simulação
                elif mode in ("live", "bci"):
                    self.mode_combo.setCurrentIndex(0)  # Hardware
                elif mode == "offline":
                    self.mode_combo.setCurrentIndex(2)  # Playback (mais próximo)

            # 3) Número de canais escolhido no Pré-Flight (8..64). Usa
            #    _set_num_channels para sincronizar TUDO (head_plot, emg_widget,
            #    combos, label, abas multimodais). Antes fazia
            #    self.num_channels = MAX_CHANNELS (=64!) o que descasava os
            #    widgets e quebrava a visualização. O valor vem direto do combo
            #    do launcher; expansion_16ch fica como compatibilidade legada.
            n_ch = choice.get("num_channels")
            if isinstance(n_ch, int) and n_ch in EXPANSION_STEPS \
                    and hasattr(self, "_set_num_channels"):
                self._set_num_channels(n_ch)
            elif choice.get("expansion_16ch") and hasattr(self, "_set_num_channels"):
                self._set_num_channels(CYTON_MAX_CHANNELS)  # legado: 16 canais

            # 4) Tipo de aquisição -> channel_signal_types
            acq = choice.get("acquisition_type", "EEG")
            if acq == "EEG":
                self.config.channel_signal_types = ["EEG"] * MAX_CHANNELS
            elif acq == "EMG":
                self.config.channel_signal_types = ["EMG"] * MAX_CHANNELS
            elif acq == "ECG":
                self.config.channel_signal_types = ["ECG"] * MAX_CHANNELS
            elif acq == "Hibrido":
                # Layout sugerido: 1-8 EEG / 9-12 EMG / 13-14 ECG / 15-16 EoG
                self.config.channel_signal_types = (
                    ["EEG"] * 8 + ["EMG"] * 4 + ["ECG"] * 2 + ["EoG"] * 2
                )[: MAX_CHANNELS]
            self.config.save()
            # Sincroniza combos da aba Filtros e Canais se já criados
            if hasattr(self, "channel_type_combos"):
                for ch, tc in enumerate(self.channel_type_combos):
                    if ch < len(self.config.channel_signal_types):
                        tc.blockSignals(True)
                        tc.setCurrentText(self.config.channel_signal_types[ch])
                        tc.blockSignals(False)
                # Atualiza filtro recomendado (label) e habilitação do threshold
                for ch in range(MAX_CHANNELS):
                    stype = self.config.channel_signal_types[ch]
                    if ch < len(self.channel_filter_hint_lbls):
                        preset = SIGNAL_FILTER_PRESETS.get(stype, SIGNAL_FILTER_PRESETS["EEG"])
                        self.channel_filter_hint_lbls[ch].setText(preset["label"])
                        self.channel_filter_hint_lbls[ch].setStyleSheet(
                            f"color: {SIGNAL_TYPE_COLORS.get(stype, COLORS['text_dim'])}; "
                            f"font-size: 9pt;")
                    if ch < len(self.channel_emg_thresh_inline):
                        self.channel_emg_thresh_inline[ch].setEnabled(stype == "EMG")
            # Reflete nas abas dependentes
            if hasattr(self, "emg_rows"):
                self._emg_refresh_channel_types()
            if hasattr(self, "ecg_channel_combo"):
                self._populate_ecg_channel_combo()
            if hasattr(self, "eog_h_combo"):
                self._populate_eog_channel_combos()
            if hasattr(self, "_joy_axes"):
                self._joy_repopulate_combos()

            # 5) Voluntário ativo (se houver)
            vol_dir = choice.get("volunteer_dir")
            if vol_dir and hasattr(self, "volunteers"):
                try:
                    self.volunteers.set_active(vol_dir)
                    if hasattr(self, "_volunteer_update_indicator"):
                        self._volunteer_update_indicator()
                except Exception:
                    pass

            # 6) Visibilidade de abas conforme o tipo de aquisição
            #    (esconde Topografia/ERP/ERS-ERD para EMG, ECG, EoG, etc.)
            self._apply_signal_mode_visibility(acq)

            # 7) Navega para a aba apropriada conforme o modo
            self._navigate_for_launcher_mode(mode, choice.get("selected_csv"))

            # 8) Modo Simulação: inicia o streaming sintético automaticamente.
            #    Antes o usuário caía na aba Conexão sem nada acontecer e achava
            #    que "a simulação não funcionava". Agora a simulação já começa e
            #    a aba Tempo Real mostra o sinal imediatamente. Usa singleShot
            #    para garantir que a janela já esteja construída/visível.
            if mode == "sim":
                QtCore.QTimer.singleShot(400, self._autostart_simulation)

            self._log(f"[Launcher] aplicado: mode={mode} acq={acq} "
                      f"port={port} canais={choice.get('num_channels') or '?'}")
        except Exception as exc:
            self._log(f"Erro aplicando choice do launcher: {exc}", error=True)

    def _navigate_for_launcher_mode(self, mode, csv_path=None):
        """Navega para a aba/sub-aba apropriada conforme o modo escolhido."""
        if not hasattr(self, "_main_tabs"): return
        # Mapas index conforme criação em _build_ui()
        nav_map = {
            "live":     (0, ("setup",   1)),   # Configurar → Conexão
            "sim":      (1, ("view",    0)),   # Visualizar → Tempo Real (sinal já visível)
            "offline":  (2, ("analyse", 1)),   # Analisar → Offline
            "bci":      (2, ("analyse", 5)),   # Analisar → Focus
            "settings": (3, ("system",  1)),   # Sistema → Configurações
        }
        target = nav_map.get(mode)
        if not target: return
        top_idx, (sub_key, sub_idx) = target
        try:
            self._main_tabs.setCurrentIndex(top_idx)
            sub = self._sub_tabs.get(sub_key)
            if sub:
                sub.setCurrentIndex(min(sub_idx, sub.count() - 1))
        except Exception:
            pass
        # Carrega CSV no Offline se foi clicado em uma sessão recente
        if mode == "offline" and csv_path and hasattr(self, "_offline_load_csv"):
            try:
                self._offline_load_csv(csv_path)
            except Exception as exc:
                self._log(f"Falha ao carregar CSV {csv_path}: {exc}", error=True)

    def _goto_realtime_view(self):
        """Leva o usuário para Visualizar → Tempo Real (onde o sinal aparece)."""
        if not hasattr(self, "_main_tabs"):
            return
        try:
            self._main_tabs.setCurrentIndex(1)          # Visualizar
            view = self._sub_tabs.get("view")
            if view:
                view.setCurrentIndex(0)                 # Tempo Real
        except Exception:
            pass

    def _autostart_simulation(self):
        """Inicia a simulação automaticamente (chamado pelo launcher).

        Garante que o combo de modo esteja em 'Simulação' e que ainda não
        haja uma thread rodando, então conecta. Idempotente: se já estiver
        conectado, apenas leva o usuário para a aba Tempo Real.
        """
        try:
            if hasattr(self, "mode_combo") and self.mode_combo.currentIndex() != 1:
                self.mode_combo.setCurrentIndex(1)      # Simulação
            already = bool(self.serial_thread and self.serial_thread.isRunning())
            if not already:
                self._connect()
            self._goto_realtime_view()
        except Exception as exc:
            self._log(f"Falha ao iniciar simulação automática: {exc}", error=True)

    def _setup_usability(self):
        """Configura: barra de status, menu Ajuda, atalhos de teclado.
        Chamada ao fim de __init__."""
        # ---- Barra de status (no rodapé) ----
        sb = self.statusBar()
        sb.setStyleSheet(
            f"QStatusBar {{ background-color: {COLORS['surface']}; "
            f"color: {COLORS['text_dim']}; "
            f"border-top: 1px solid {COLORS['border']}; }}")
        self.status_state_lbl = QtWidgets.QLabel("Pronto.")
        self.status_state_lbl.setStyleSheet(
            f"color: {COLORS['text_dim']}; padding: 0 8px;")
        sb.addWidget(self.status_state_lbl, stretch=1)
        version_lbl = QtWidgets.QLabel(
            f"{APP_NAME} v{APP_VERSION}")
        version_lbl.setStyleSheet(
            f"color: {COLORS['text_dim']}; padding: 0 8px;")
        sb.addPermanentWidget(version_lbl)

        # ---- Menu Ajuda ----
        menubar = self.menuBar()
        menubar.setStyleSheet(
            f"QMenuBar {{ background-color: {COLORS['surface']}; "
            f"color: {COLORS['text']}; border-bottom: 1px solid {COLORS['border']}; }}"
            f"QMenuBar::item:selected {{ background-color: {COLORS['accent_dim']}; "
            f"color: {COLORS['background']}; }}"
            f"QMenu {{ background-color: {COLORS['surface_alt']}; "
            f"color: {COLORS['text']}; border: 1px solid {COLORS['border']}; }}"
            f"QMenu::item:selected {{ background-color: {COLORS['accent_dim']}; "
            f"color: {COLORS['background']}; }}")
        tools_menu = menubar.addMenu("Ferramentas")
        act_prof_exp = tools_menu.addAction("Exportar perfil de protocolo...")
        act_prof_exp.triggered.connect(self._export_protocol_profile)
        act_prof_imp = tools_menu.addAction("Importar perfil de protocolo...")
        act_prof_imp.triggered.connect(self._import_protocol_profile)
        tools_menu.addSeparator()
        act_maker = tools_menu.addAction("Área Maker (receitas de análise)...")
        act_maker.triggered.connect(self._open_recipe)
        act_logs = tools_menu.addAction("Abrir pasta de logs")
        act_logs.triggered.connect(self._open_logs_dir)
        help_menu = menubar.addMenu("Ajuda")
        act_help = help_menu.addAction("Assistente de ajuda...")
        act_help.setShortcut("F1")
        act_help.triggered.connect(self._show_help_assistant)
        help_menu.addSeparator()
        act_about = help_menu.addAction("Sobre o aplicativo...")
        act_about.triggered.connect(self._show_about_dialog)
        act_shortcuts = help_menu.addAction("Atalhos de teclado")
        act_shortcuts.triggered.connect(self._show_shortcuts_dialog)
        help_menu.addSeparator()
        act_docs = help_menu.addAction("Pasta de configuração / sessões")
        act_docs.triggered.connect(self._open_save_dir)
        act_terms = help_menu.addAction("Termo de uso e privacidade...")
        act_terms.triggered.connect(self._show_terms_dialog)
        help_menu.addSeparator()
        act_update = help_menu.addAction("Verificar atualizações...")
        act_update.triggered.connect(self._check_updates_manual)
        act_diag = help_menu.addAction("Diagnóstico de erros (simular)...")
        act_diag.triggered.connect(self._show_error_diagnostics)

        # ---- Atalhos de teclado ----
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self,
                         activated=self._toggle_recording)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+C"), self,
                         activated=self._toggle_connection)
        QtGui.QShortcut(QtGui.QKeySequence("F5"), self,
                         activated=self._refresh_ports)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Q"), self,
                         activated=self.close)
        # Comando palette + screenshot + modo apresentação
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+P"), self,
                         activated=self._show_command_palette)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+P"), self,
                         activated=self._screenshot_current_tab)
        QtGui.QShortcut(QtGui.QKeySequence("F11"), self,
                         activated=self._toggle_presentation_mode)

    def _screenshot_current_tab(self):
        """Captura aba atual como PNG, salvo em sessions/screenshots/."""
        try:
            # Captura a janela toda (ou só o widget central)
            central = self.centralWidget() or self
            pixmap = central.grab()
            # Pasta de destino
            shots_dir = os.path.join(self.config.save_directory, "screenshots")
            os.makedirs(shots_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Nome da aba atual (para nome do arquivo)
            tab_name = "tab"
            try:
                top_i = self._main_tabs.currentIndex()
                top_name = self._main_tabs.tabText(top_i).replace(" ", "_")
                # Pega sub-aba atual
                sub = self._main_tabs.widget(top_i)
                if isinstance(sub, QtWidgets.QTabWidget):
                    sub_name = sub.tabText(sub.currentIndex()).replace(" ", "_")
                    tab_name = f"{top_name}__{sub_name}"
                else:
                    tab_name = top_name
            except Exception: pass
            out_path = os.path.join(shots_dir, f"screenshot_{tab_name}_{ts}.png")
            if pixmap.save(out_path, "PNG"):
                self._log(f"Screenshot salvo: {out_path}")
                self._audit_event("screenshot_captured", path=out_path)
                # Notifica via status bar
                if hasattr(self, "status_state_lbl"):
                    self.status_state_lbl.setText(f"Screenshot: {os.path.basename(out_path)}")
        except Exception as exc:
            self._log(f"Falha ao capturar screenshot: {exc}", error=True)

    def _toggle_presentation_mode(self):
        """Alterna modo apresentação: fullscreen + esconde menu/status bar."""
        self._presentation_mode = not getattr(self, "_presentation_mode", False)
        if self._presentation_mode:
            self._pre_pres_state = self.windowState()
            self.showFullScreen()
            if hasattr(self, "menuBar"):
                self.menuBar().setVisible(False)
            sb = self.statusBar() if hasattr(self, "statusBar") else None
            if sb: sb.setVisible(False)
            if hasattr(self, "header_widget"):
                self.header_widget.setVisible(False)
            self._log("Modo apresentação ATIVADO (F11 para sair).")
        else:
            self.setWindowState(QtCore.Qt.WindowState.WindowNoState)
            self.showNormal()
            if hasattr(self, "menuBar"):
                self.menuBar().setVisible(True)
            sb = self.statusBar() if hasattr(self, "statusBar") else None
            if sb: sb.setVisible(True)
            if hasattr(self, "header_widget"):
                self.header_widget.setVisible(True)
            self._log("Modo apresentação desativado.")

    def _show_command_palette(self):
        """Abre paleta de comandos (Ctrl+Shift+P) — busca global de abas/comandos."""
        commands = self._build_palette_commands()
        dlg = _CommandPalette(commands, parent=self)
        # Centraliza sobre a janela principal
        geo = self.geometry()
        dlg.move(geo.x() + (geo.width() - dlg.width()) // 2,
                  geo.y() + 80)
        dlg.exec()

    def _build_palette_commands(self):
        """Constrói a lista de comandos para a paleta."""
        cmds = []
        # ---- Abas top-level ----
        if hasattr(self, "_main_tabs") and hasattr(self, "_sub_tabs"):
            top_titles = ["Configurar", "Visualizar", "Analisar", "Sistema"]
            for i, title in enumerate(top_titles):
                if i < self._main_tabs.count():
                    cmds.append({
                        "label": tr(title), "category": "Grupo",
                        "action": (lambda idx=i: self._main_tabs.setCurrentIndex(idx)),
                    })
            # Sub-abas
            sub_specs = [
                ("setup",   ["Voluntários", "Conexão", "Filtros e Canais",
                              "Hardware", "Calibração"]),
                ("view",    ["Tempo Real", "Topografia", "Espectrograma",
                              "Bio (EMG/ECG/EoG)", "Histórico", "Layout Custom"]),
                ("analyse", ["Análises", "Offline", "ERP", "Conectividade",
                              "ERS/ERD", "Focus / SSVEP", "EMG Joystick"]),
                ("system",  ["Rede e Eventos", "Configurações"]),
            ]
            top_map = {"setup": 0, "view": 1, "analyse": 2, "system": 3}
            for key, titles in sub_specs:
                sub = self._sub_tabs.get(key); top_i = top_map[key]
                if not sub: continue
                for i, t in enumerate(titles):
                    if i >= sub.count(): break
                    cmds.append({
                        "label": tr(t), "category": "Aba",
                        "action": (lambda ti=top_i, si=i, sub_w=sub:
                                   (self._main_tabs.setCurrentIndex(ti),
                                    sub_w.setCurrentIndex(si))),
                    })
        # ---- Comandos de hardware/gravação ----
        common = [
            ("Conectar/Desconectar (Ctrl+Shift+C)", "Comando", self._toggle_connection),
            ("Iniciar/Parar gravação (Ctrl+R)",     "Comando", self._toggle_recording),
            ("Atualizar lista de portas (F5)",      "Comando", self._refresh_ports),
            ("Captura de tela da aba (Ctrl+P)",     "Comando", self._screenshot_current_tab),
            ("Modo apresentação (F11)",             "Comando", self._toggle_presentation_mode),
            ("Abrir pasta de sessões",              "Comando", self._open_save_dir),
            ("Sobre o aplicativo",                  "Comando", self._show_about_dialog),
            ("Atalhos de teclado",                  "Comando", self._show_shortcuts_dialog),
            ("Detectar canais ruins",               "Comando", self._detect_bad_channels),
            ("Exportar para EDF",                   "Comando", self._export_to_edf),
            ("Exportar para FIF (MNE) + script",    "Comando", self._export_to_fif),
            ("Exportar BIDS-EEG",                   "Comando", self._export_to_bids),
            ("Relatório PDF",                       "Comando", self._export_pdf_report),
        ]
        for label, cat, action in common:
            if action is not None:
                cmds.append({"label": label, "category": cat, "action": action})
        # ---- Voluntários ----
        try:
            for prof in self.volunteers.list_volunteers():
                vid = prof.get("vid", "?")
                nome = prof.get("nome", "")
                cmds.append({
                    "label": f"Selecionar voluntário: {vid} — {nome}",
                    "category": "Voluntário",
                    "action": (lambda dn=prof.get("_dirname"):
                               (self.volunteers.set_active(dn),
                                self._volunteer_update_indicator())),
                })
        except Exception:
            pass
        return cmds

    def _show_about_dialog(self):
        """Diálogo Sobre — versão, autores, créditos."""
        text = (
            f"<h2 style='color:{COLORS['accent']}; margin-bottom:4px;'>"
            f"{APP_NAME}</h2>"
            f"<p style='color:{COLORS['text_dim']}; margin-top:0;'>"
            f"{APP_EDITION} · versão <b>{APP_VERSION}</b></p>"
            f"<p>Aplicativo 100% Python para coleta, visualização e análise "
            f"de sinais EEG em tempo real e em modo offline.</p>"
            f"<p><b>Recursos principais:</b></p>"
            f"<ul style='margin-top:0;'>"
            f"<li>Aquisição 8/16 canais via OpenBCI Cyton (Hardware/Simulação/Playback)</li>"
            f"<li>Análises: FFT, bandas, espectrograma, head plot, ERP, conectividade</li>"
            f"<li>Análise ERS/ERD com formato compatível</li>"
            f"<li>Cadastro de voluntários e organização por sujeito</li>"
            f"<li>Exportação EDF, FIF (MNE), PDF · Streaming LSL/UDP</li>"
            f"<li>Audit log + SHA-256 para validação de pesquisa</li>"
            f"</ul>"
            f"<p>Desenvolvido por <b>{APP_AUTHORS}</b> · © {APP_YEAR}</p>"
            f"<p style='color:{COLORS['text_dim']}; font-size:9pt;'>"
            f"Software de pesquisa. NÃO é dispositivo médico aprovado por FDA/ANVISA/CE."
            f"</p>"
        )
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle(f"Sobre o {APP_NAME}")
        msg.setTextFormat(QtCore.Qt.TextFormat.RichText)
        msg.setText(text)
        msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        msg.exec()

    def _show_shortcuts_dialog(self):
        """Lista os atalhos de teclado disponíveis."""
        text = (
            f"<h3 style='color:{COLORS['accent']}; margin-bottom:8px;'>"
            f"Atalhos de teclado</h3>"
            f"<table cellpadding='6' style='font-family:{FONT_DATA_STACK};'>"
            f"<tr><td><b>F1</b></td><td>Sobre o aplicativo</td></tr>"
            f"<tr><td><b>F5</b></td><td>Atualizar lista de portas COM</td></tr>"
            f"<tr><td><b>Ctrl+R</b></td><td>Iniciar / parar gravação</td></tr>"
            f"<tr><td><b>Ctrl+Shift+C</b></td><td>Conectar / desconectar</td></tr>"
            f"<tr><td><b>Ctrl+Q</b></td><td>Fechar o aplicativo</td></tr>"
            f"<tr><td><b>M</b></td><td>Injetar marker rápido (label 'M')</td></tr>"
            f"</table>"
            f"<p style='color:{COLORS['text_dim']}; font-size:9pt;'>"
            f"Arrastar e soltar um arquivo <code>.csv</code> sobre a janela "
            f"abre-o automaticamente no modo Offline.</p>"
        )
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Atalhos de teclado")
        msg.setTextFormat(QtCore.Qt.TextFormat.RichText)
        msg.setText(text)
        msg.exec()

    def _open_save_dir(self):
        """Abre o diretório de salvamento no explorador de arquivos do SO."""
        path = self.config.save_directory
        try:
            os.makedirs(path, exist_ok=True)
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                import subprocess; subprocess.Popen(["open", path])
            else:
                import subprocess; subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Abrir pasta", str(exc))

    def _open_logs_dir(self):
        """Abre a pasta de logs (Documentos/EEG_Coletor/logs)."""
        path = os.path.join(DOC_DIR, "logs")
        try:
            os.makedirs(path, exist_ok=True)
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                import subprocess; subprocess.Popen(["open", path])
            else:
                import subprocess; subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Abrir pasta", str(exc))

    def _export_protocol_profile(self):
        """Exporta a montagem/setup (mapeamento, tipos de sinal, EMG, impedância)
        como um perfil .json reutilizável entre máquinas."""
        prof = {
            "profile_version": "1.0",
            "channel_mapping": self.config.channel_mapping,
            "channel_signal_types": self.config.channel_signal_types,
            "emg_threshold_uV": self.config.emg_threshold_uV,
            "emg_channel_muscle": self.config.emg_channel_muscle,
            "emg_channel_mvc_uV": self.config.emg_channel_mvc_uV,
            "emg_envelope_method": self.config.emg_envelope_method,
            "emg_envelope_window_ms": self.config.emg_envelope_window_ms,
            "imp_good_max": self.config.imp_good_max,
            "imp_acceptable_max": self.config.imp_acceptable_max,
        }
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exportar perfil de protocolo", "perfil_protocolo.json",
            "Perfil (*.json)")
        if not p:
            return
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(prof, f, ensure_ascii=False, indent=2)
            QtWidgets.QMessageBox.information(self, "Perfil de protocolo",
                                              f"Perfil salvo:\n{p}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Perfil de protocolo",
                                          f"Falha ao salvar: {exc}")

    def _import_protocol_profile(self):
        """Importa um perfil de protocolo .json (validação defensiva), salva no
        config e pede para reabrir para aplicar tudo."""
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Importar perfil de protocolo", "", "Perfil (*.json)")
        if not p:
            return
        try:
            with open(p, encoding="utf-8") as f:
                prof = json.load(f)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Perfil de protocolo",
                                          f"Falha ao ler: {exc}")
            return
        cm = prof.get("channel_mapping")
        if isinstance(cm, list) and len(cm) == len(self.config.channel_mapping):
            self.config.channel_mapping = cm
        for key in ("channel_signal_types", "emg_threshold_uV",
                    "emg_channel_muscle", "emg_channel_mvc_uV"):
            v = prof.get(key)
            if isinstance(v, list):
                setattr(self.config, key, v)
        if isinstance(prof.get("emg_envelope_method"), str):
            self.config.emg_envelope_method = prof["emg_envelope_method"]
        for key in ("emg_envelope_window_ms", "imp_good_max", "imp_acceptable_max"):
            try:
                setattr(self.config, key,
                        type(getattr(self.config, key))(prof[key]))
            except Exception:
                pass
        self.config.save()
        QtWidgets.QMessageBox.information(
            self, "Perfil de protocolo",
            "Perfil importado e salvo. Reabra o programa para aplicar totalmente "
            "(mapeamento e tipos de sinal).")

    def _show_help_assistant(self):
        """Abre o assistente de ajuda OFFLINE (chat de FAQ + erros, sem internet)."""
        HelpAssistantDialog(self).exec()

    def _show_error_diagnostics(self):
        """Abre o catálogo de erros para consultar/simular as notificações."""
        ErrorDiagnosticsDialog(self).exec()

    def _show_terms_dialog(self):
        """Mostra o Termo de Uso e Privacidade (somente leitura)."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Termo de uso e privacidade")
        dlg.setMinimumSize(680, 600)
        v = QtWidgets.QVBoxLayout(dlg)
        br = QtWidgets.QTextBrowser(); br.setOpenExternalLinks(True)
        txt = _load_terms_text()
        try: br.setMarkdown(txt)
        except Exception: br.setPlainText(txt)
        v.addWidget(br, 1)
        if getattr(self.config, "terms_accepted", False):
            v.addWidget(QtWidgets.QLabel(
                f"<i>Aceito (versão {self.config.terms_version or '?'}) em "
                f"{self.config.terms_accepted_at or '?'} — registro local.</i>"))
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(dlg.reject); bb.accepted.connect(dlg.accept)
        v.addWidget(bb)
        try: dlg.setStyleSheet(build_stylesheet(COLORS))
        except Exception: pass
        dlg.exec()

    def _check_updates_manual(self):
        """Verificação MANUAL e opcional de atualização. Privacidade: só BAIXA
        código do GitHub, NUNCA envia dados. Offline -> mensagem amigável."""
        import json as _json, ssl as _ssl, hashlib as _hashlib, shutil as _shutil
        import urllib.request as _ur
        cfg_path = os.path.join(SCRIPT_DIR, "update_config.json")
        py_path  = os.path.join(SCRIPT_DIR, "EEG_Data_Collector.py")
        cfg = {}
        try:
            if os.path.exists(cfg_path):
                with open(cfg_path, encoding="utf-8") as f: cfg = _json.load(f)
        except Exception:
            cfg = {}
        url = (cfg or {}).get("version_url", "")
        if not url or "SEU_USUARIO" in url:
            QtWidgets.QMessageBox.information(self, "Atualizações",
                "A atualização automática não está configurada nesta instalação.\n\n"
                "Este software funciona offline e não envia nenhum dado. "
                f"O código-fonte está disponível em:\n{CODE_URL}")
            return

        def _vt(v):
            try: return tuple(int(x) for x in str(v).split("."))
            except Exception: return (0,)

        def _get(u, t):
            ctx = _ssl.create_default_context()
            req = _ur.Request(u, headers={"User-Agent": "EEG-Collector-Updater"})
            with _ur.urlopen(req, timeout=t, context=ctx) as r:
                return r.read()

        try:
            manifest = _json.loads(_get(url, 6).decode("utf-8"))
        except Exception as exc:
            logging.getLogger("eeg").info("verificação manual de update falhou: %s", exc)
            QtWidgets.QMessageBox.information(self, "Atualizações",
                "Não foi possível verificar atualizações agora.\n\n"
                "Você pode estar offline — o que é normal: o programa funciona "
                "100% offline. Tente novamente quando tiver internet.")
            return
        if _vt(manifest.get("version", "0")) <= _vt(APP_VERSION):
            QtWidgets.QMessageBox.information(self, "Atualizações",
                f"Você já está na versão mais recente ({APP_VERSION}).")
            return
        novo = manifest.get("version", "?")
        changelog = (manifest.get("changelog") or "").strip()
        if QtWidgets.QMessageBox.question(self, "Atualização disponível",
                f"Nova versão {novo} disponível (você tem {APP_VERSION}).\n\n"
                f"{changelog}\n\nBaixar e aplicar agora? Apenas o código é "
                "baixado; nenhum dado seu é enviado.") \
                != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            blob = _get(manifest["py_url"], 60)
            expected = (manifest.get("sha256") or "").lower().strip()
            got = _hashlib.sha256(blob).hexdigest().lower()
            if expected and got != expected:
                raise ValueError("SHA-256 não confere (download corrompido/alterado).")
            tmp = py_path + ".new"
            with open(tmp, "wb") as f: f.write(blob)
            try: _shutil.copyfile(py_path, py_path + ".bak")
            except Exception: pass
            os.replace(tmp, py_path)                 # troca atômica
        except Exception as exc:
            self._notify_error("E401" if "SHA-256" in str(exc) else "E403",
                               str(exc), exc=exc)
            return
        QtWidgets.QMessageBox.information(self, "Atualização aplicada",
            f"Atualizado para a versão {novo}.\n\n"
            "Feche e abra o aplicativo para usar a nova versão.")

    def _update_status_state(self, text=None):
        """Atualiza a barra de status do rodapé com o estado atual."""
        if not hasattr(self, "status_state_lbl"): return
        if text is None:
            # Auto-detecta o estado
            if self.is_recording:
                text = f"Gravando sessão: {self.session_name}"
                color = COLORS["error"]
            elif self.serial_thread and self.serial_thread.isRunning():
                mode = self.serial_thread.mode
                if mode == "simulation":
                    text = "ATENÇÃO: modo SIMULAÇÃO ativo — sinais não são reais"
                elif mode == "playback":
                    text = "Conectado em modo PLAYBACK (replay de CSV)"
                else:
                    text = f"Conectado: {self.serial_thread.port}"
                color = COLORS["accent"]
            else:
                text = "Pronto."
                color = COLORS["text_dim"]
        else:
            color = COLORS["text_dim"]
        self.status_state_lbl.setText(text)
        self.status_state_lbl.setStyleSheet(
            f"color: {color}; padding: 0 8px;")
        # Atualiza também o banner de modo no header (se houver)
        self._update_mode_banner()

    def _update_mode_banner(self):
        """Mostra/esconde banner amarelo de aviso no header para Simulação/Playback."""
        if not hasattr(self, "mode_banner_label"): return
        if self.serial_thread and self.serial_thread.isRunning():
            mode = self.serial_thread.mode
            if mode == "simulation":
                self.mode_banner_label.setText("MODO SIMULAÇÃO")
                self.mode_banner_label.setStyleSheet(
                    f"background-color: {COLORS['warning']}; "
                    f"color: {COLORS['background']}; "
                    f"padding: 2px 10px; font-weight: bold; "
                    f"border-radius: 3px;")
                self.mode_banner_label.setVisible(True)
                return
            elif mode == "playback":
                self.mode_banner_label.setText("MODO PLAYBACK")
                self.mode_banner_label.setStyleSheet(
                    f"background-color: {COLORS['expansion']}; "
                    f"color: {COLORS['background']}; "
                    f"padding: 2px 10px; font-weight: bold; "
                    f"border-radius: 3px;")
                self.mode_banner_label.setVisible(True)
                return
        self.mode_banner_label.setVisible(False)

    def _update_window_title(self):
        """Atualiza o título da janela conforme estado (gravando? modo?)."""
        base = f"{APP_NAME} — {APP_EDITION} v{APP_VERSION}"
        if self.is_recording and self.session_name:
            self.setWindowTitle(f"[GRAVANDO]  {self.session_name}  ·  {base}")
        elif self.serial_thread and self.serial_thread.isRunning():
            mode = self.serial_thread.mode
            if mode == "simulation":
                self.setWindowTitle(f"[SIMULAÇÃO]  {base}")
            elif mode == "playback":
                self.setWindowTitle(f"[PLAYBACK]  {base}")
            else:
                self.setWindowTitle(f"[CONECTADO]  {base}")
        else:
            self.setWindowTitle(base)

    # ---- Drag and drop de arquivos CSV ----
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(".csv"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(".csv"):
                # Abre no modo Offline
                try:
                    self._offline_load_csv(path)
                    # Navega para a aba Offline (Analisar > Offline)
                    if hasattr(self, "_main_tabs"):
                        self._main_tabs.setCurrentIndex(2)
                        if hasattr(self, "_sub_tabs") and "analyse" in self._sub_tabs:
                            self._sub_tabs["analyse"].setCurrentIndex(1)
                    self._update_status_state(f"Arquivo aberto: {os.path.basename(path)}")
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(self, "Arquivo",
                        f"Não foi possível abrir o arquivo:\n{path}\n\n"
                        f"Detalhe técnico: {exc}")
                event.acceptProposedAction()
                return
        event.ignore()

    def closeEvent(self, event):
        if self.is_recording: self._stop_recording()
        if self.serial_thread: self.serial_thread.stop()
        if hasattr(self, "imp_test_running") and self.imp_test_running:
            self._stop_impedance_test()
        self.udp.stop()
        self.lsl.stop()
        self._audit_event("app_close")
        self._close_audit_log()
        self.config.save()
        event.accept()

    # ==================================================================
    # Audit log estruturado (events.jsonl)
    # ==================================================================
    def _open_audit_log(self):
        """Abre/cria events.jsonl na pasta da sessão atual (anexar)."""
        if self._audit_fp is not None:
            return
        target_dir = self.current_session_dir or self.config.save_directory
        try:
            os.makedirs(target_dir, exist_ok=True)
            path = os.path.join(target_dir, "events.jsonl")
            self._audit_fp = open(path, "a", encoding="utf-8")
        except Exception as exc:
            print(f"[audit] falha abrindo events.jsonl: {exc}")
            self._audit_fp = None

    def _close_audit_log(self):
        if self._audit_fp is not None:
            try: self._audit_fp.close()
            except Exception: pass
            self._audit_fp = None

    def _audit_event(self, action, **detail):
        """Registra uma ação do operador como linha JSON em events.jsonl.
        Inclui sempre: iso timestamp, action, sessão atual, num_channels."""
        if self._audit_fp is None:
            self._open_audit_log()
        if self._audit_fp is None:
            return
        try:
            rec = {
                "t":           datetime.now().isoformat(timespec="milliseconds"),
                "action":      action,
                "session":     self.session_name or "(none)",
                "subject":     self.config.subject,
                "num_channels":self.num_channels,
                "recording":   self.is_recording,
            }
            if detail:
                rec["detail"] = detail
            self._audit_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._audit_fp.flush()
        except Exception as exc:
            print(f"[audit] falha escrevendo: {exc}")

    # ==================================================================
    # Fontes nos plots do pyqtgraph
    # ==================================================================
    def _apply_plot_fonts(self):
        """Aplica JetBrains Mono nos tick labels (numericos) e Inter nos
        labels descritivos dos eixos de TODOS os PlotItems da janela."""
        tick_font  = QtGui.QFont(FONT_DATA, 9)
        label_font = QtGui.QFont(FONT_UI, 10)

        def style_plot_item(plot_item):
            try:
                for axis_name in ("bottom", "left", "top", "right"):
                    ax = plot_item.getAxis(axis_name)
                    if ax is None: continue
                    try:
                        ax.setStyle(tickFont=tick_font)
                    except Exception: pass
                    try:
                        if hasattr(ax, "label") and ax.label is not None:
                            ax.label.setFont(label_font)
                    except Exception: pass
            except Exception: pass

        # Plots individuais
        plot_widgets = []
        for attr in ("fft_plot", "band_plot", "spec_widget", "accel_plot",
                     "history_plot"):
            w = getattr(self, attr, None)
            if w is not None:
                plot_widgets.append(w)
        for w in plot_widgets:
            try:
                style_plot_item(w.getPlotItem())
            except Exception: pass

        # 8/16 canais de tempo real
        if hasattr(self, "channel_plots"):
            for p in self.channel_plots:
                style_plot_item(p)

        # Mini-plot do FocusMeter (se ja existir)
        if hasattr(self, "focus_meter") and hasattr(self.focus_meter, "plot"):
            try:
                style_plot_item(self.focus_meter.plot.getPlotItem())
            except Exception: pass
        if hasattr(self, "emg_widget") and hasattr(self.emg_widget, "plot"):
            try:
                style_plot_item(self.emg_widget.plot.getPlotItem())
            except Exception: pass

        # Plots dos slots do Layout Custom
        if hasattr(self, "layout_slots"):
            for slot in self.layout_slots:
                w = slot.get("widget")
                if w is not None and hasattr(w, "getPlotItem"):
                    try:
                        style_plot_item(w.getPlotItem())
                    except Exception: pass

    # ==================================================================
    # Aba CALIBRACAO — teste de impedância dedicado
    # ==================================================================
    def _build_calibration_tab(self):
        """Aba Calibração — desenhada para mostrar TODOS os 16 canais sem rolagem."""
        widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(10, 8, 10, 8); outer.setSpacing(6)

        # Header compacto em UMA linha
        header = QtWidgets.QLabel(
            f"<span style='color:{COLORS['accent']}; font-size:14pt; font-weight:bold;'>"
            "Calibração da Touca</span>"
            f"  <span style='color:{COLORS['text_dim']};'>"
            "Injeta corrente de teste (6 nA @ 31.25 Hz) e mede a impedância via FFT."
            "</span>"
        )
        header.setTextFormat(QtCore.Qt.TextFormat.RichText)
        outer.addWidget(header)

        # Linha unica: limites + controles + status (compacto)
        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.setSpacing(8)
        ctrl_row.addWidget(QtWidgets.QLabel("Verde até:"))
        self.imp_good_spin = QtWidgets.QDoubleSpinBox()
        self.imp_good_spin.setRange(0.1, 999.0); self.imp_good_spin.setDecimals(1)
        self.imp_good_spin.setSingleStep(1.0); self.imp_good_spin.setSuffix(" kΩ")
        self.imp_good_spin.setMaximumWidth(110)
        self.imp_good_spin.setValue(self.config.imp_good_max)
        self.imp_good_spin.valueChanged.connect(self._on_imp_thresholds_changed)
        ctrl_row.addWidget(self.imp_good_spin)

        ctrl_row.addSpacing(8)
        ctrl_row.addWidget(QtWidgets.QLabel("Amarelo até:"))
        self.imp_acc_spin = QtWidgets.QDoubleSpinBox()
        self.imp_acc_spin.setRange(0.1, 9999.0); self.imp_acc_spin.setDecimals(1)
        self.imp_acc_spin.setSingleStep(5.0); self.imp_acc_spin.setSuffix(" kΩ")
        self.imp_acc_spin.setMaximumWidth(120)
        self.imp_acc_spin.setValue(self.config.imp_acceptable_max)
        self.imp_acc_spin.valueChanged.connect(self._on_imp_thresholds_changed)
        ctrl_row.addWidget(self.imp_acc_spin)

        ctrl_row.addSpacing(20)
        self.imp_start_btn = QtWidgets.QPushButton("▶  Iniciar Teste")
        self.imp_start_btn.setMinimumHeight(34)
        self.imp_start_btn.clicked.connect(self._start_impedance_test)
        ctrl_row.addWidget(self.imp_start_btn)
        self.imp_stop_btn = QtWidgets.QPushButton("■  Parar")
        self.imp_stop_btn.setMinimumHeight(34)
        self.imp_stop_btn.clicked.connect(self._stop_impedance_test)
        self.imp_stop_btn.setEnabled(False)
        ctrl_row.addWidget(self.imp_stop_btn)

        ctrl_row.addSpacing(10)
        tips_btn = QtWidgets.QPushButton("Dicas")
        tips_btn.setToolTip("Como reduzir impedância dos eletrodos")
        tips_btn.clicked.connect(self._show_impedance_tips)
        ctrl_row.addWidget(tips_btn)
        ctrl_row.addStretch()

        self.imp_status_label = QtWidgets.QLabel("Pronto. Conecte em modo Hardware para testar.")
        self.imp_status_label.setStyleSheet(f"color: {COLORS['text_dim']};")
        ctrl_row.addWidget(self.imp_status_label)
        outer.addLayout(ctrl_row)

        # Legenda em UMA linha (substitui a info box grande)
        self.imp_info_label = QtWidgets.QLabel("")
        self.imp_info_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.imp_info_label.setWordWrap(False)
        self.imp_info_label.setStyleSheet(
            f"color: {COLORS['text_dim']}; background-color: {COLORS['surface_alt']}; "
            f"padding: 4px 8px; border: 1px solid {COLORS['border']}; border-radius: 3px;")
        outer.addWidget(self.imp_info_label)
        self._refresh_imp_info()

        # Tabela de impedância dimensionada para mostrar TODOS os 16 canais
        self.imp_table = QtWidgets.QTableWidget(MAX_CHANNELS, 5)
        self.imp_table.setHorizontalHeaderLabels(
            ["Canal", "Eletrodo", "Amplitude (µV)", "Impedância (kΩ)", "Qualidade"]
        )
        self.imp_table.verticalHeader().setVisible(False)
        self.imp_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.imp_table.setAlternatingRowColors(True)
        self.imp_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        # Alturas compactas para caber tudo
        ROW_H = 26
        self.imp_table.verticalHeader().setDefaultSectionSize(ROW_H)
        self.imp_table.horizontalHeader().setFixedHeight(28)
        for ch in range(MAX_CHANNELS):
            ch_item = QtWidgets.QTableWidgetItem(f"CH{ch + 1}")
            ch_item.setForeground(QtGui.QColor(CHANNEL_COLORS[ch]))
            ch_item.setFont(QtGui.QFont(FONT_DATA, 10, QtGui.QFont.Weight.Bold))
            self.imp_table.setItem(ch, 0, ch_item)
            for col in range(1, 5):
                it = QtWidgets.QTableWidgetItem("—")
                it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.imp_table.setItem(ch, col, it)
        # Altura exata: 16 linhas + header + bordas
        exact_h = ROW_H * MAX_CHANNELS + 28 + 4
        self.imp_table.setMinimumHeight(exact_h)
        self.imp_table.setMaximumHeight(exact_h + 40)
        # Sem barras de rolagem na tabela em si
        self.imp_table.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.imp_table.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(self.imp_table, stretch=1)
        return widget

    def _show_impedance_tips(self):
        QtWidgets.QMessageBox.information(
            self, "Dicas para reduzir impedância",
            "• Limpe a área da pele com algodão + álcool antes da touca\n"
            "• Se o eletrodo estiver seco: hidrate com gel/pasta condutiva (NaCl)\n"
            "• Verifique se o cabelo não está isolando o eletrodo\n"
            "• Aguarde 2-3 minutos após colocar gel (impedância diminui com o tempo)\n"
            "• Eletrodos de referência (DRL/REF) são especialmente críticos\n\n"
            "Faixa típica de boa qualidade EEG: < 10 kΩ.\n"
            "Sistemas modernos (active electrodes) toleram até 50 kΩ."
        )

    # ==================================================================
    # Aba LAYOUT CUSTOM — grid de paineis selecionaveis
    # ==================================================================
    PANEL_KINDS = [
        ("(vazio)",              "empty"),
        ("Tempo Real (1 canal)", "ts1"),
        ("FFT",                  "fft"),
        ("Bandas EEG",           "bands"),
        ("Head Plot",            "head"),
        ("Espectrograma",        "spec"),
        ("Acelerômetro",         "accel"),
        ("Focus (β/(α+β))",      "focus"),
    ]

    def _build_layout_tab(self):
        widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(10, 10, 10, 10); outer.setSpacing(8)

        info = QtWidgets.QLabel(
            "<b>Layout customizavel:</b> 4 paineis ajustaveis. "
            "<b>Arraste as bordas</b> (linhas entre os paineis) para redimensionar. "
            "Use <i>Salvar layout</i> para guardar as escolhas (tipo de painel, canal "
            "e tamanhos) — sera restaurado na próxima abertura."
        )
        info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        info.setStyleSheet(f"color: {COLORS['text_dim']}; padding: 4px;")
        info.setWordWrap(True)
        outer.addWidget(info)

        # Barra de ações
        toolbar = QtWidgets.QHBoxLayout()
        save_btn = QtWidgets.QPushButton("Salvar layout")
        save_btn.clicked.connect(self._save_layout_config)
        toolbar.addWidget(save_btn)
        reset_btn = QtWidgets.QPushButton("↺ Restaurar padrão")
        reset_btn.clicked.connect(self._reset_layout_to_default)
        toolbar.addWidget(reset_btn)
        toolbar.addStretch()
        outer.addLayout(toolbar)

        # SPLITTERS — 4 paineis ajustaveis (2x2)
        # outer split (horizontal): col esquerda | col direita
        # cada col split (vertical): top | bottom
        self.layout_slots = []
        cfg = self.config.layout_slots_cfg

        self.left_col_split  = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.right_col_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.left_col_split.setChildrenCollapsible(False)
        self.right_col_split.setChildrenCollapsible(False)
        self.left_col_split.setHandleWidth(6)
        self.right_col_split.setHandleWidth(6)

        # Slot 0: top-left | Slot 1: top-right | Slot 2: bottom-left | Slot 3: bottom-right
        for idx in range(4):
            item = cfg[idx] if (idx < len(cfg) and isinstance(cfg[idx], dict)) else {}
            kind = item.get("kind", "empty")
            ch   = item.get("channel", 0)
            slot = self._create_layout_slot(idx, kind, ch)
            target = self.left_col_split if (idx % 2 == 0) else self.right_col_split
            target.addWidget(slot["frame"])

        # Restaura tamanhos
        try:
            self.left_col_split.setSizes(self.config.layout_split_left)
            self.right_col_split.setSizes(self.config.layout_split_right)
        except Exception: pass

        self.main_layout_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.main_layout_split.setChildrenCollapsible(False)
        self.main_layout_split.setHandleWidth(6)
        self.main_layout_split.addWidget(self.left_col_split)
        self.main_layout_split.addWidget(self.right_col_split)
        try:
            self.main_layout_split.setSizes(self.config.layout_split_h_sizes)
        except Exception: pass

        outer.addWidget(self.main_layout_split, stretch=1)
        return widget

    def _save_layout_config(self):
        if not hasattr(self, "layout_slots"): return
        cfg = []
        for slot in self.layout_slots:
            cfg.append({
                "kind":    slot["kind"],
                "channel": slot["ch_combo"].currentIndex(),
            })
        self.config.layout_slots_cfg     = cfg
        self.config.layout_split_h_sizes = self.main_layout_split.sizes()
        self.config.layout_split_left    = self.left_col_split.sizes()
        self.config.layout_split_right   = self.right_col_split.sizes()
        self.config.save()
        self._log("Layout customizado salvo em config.json")

    def _reset_layout_to_default(self):
        defaults = ["ts1", "fft", "head", "bands"]
        for idx, slot in enumerate(self.layout_slots):
            kind = defaults[idx]
            kind_idx = next((i for i, (_, c) in enumerate(self.PANEL_KINDS) if c == kind), 0)
            slot["combo"].blockSignals(True)
            slot["combo"].setCurrentIndex(kind_idx)
            slot["combo"].blockSignals(False)
            slot["ch_combo"].setCurrentIndex(0)
            self._set_slot_kind(slot, kind)
        # Tamanhos iguais
        total = max(800, self.width())
        self.main_layout_split.setSizes([total // 2, total // 2])
        h = max(400, self.height() - 200)
        self.left_col_split.setSizes([h // 2, h // 2])
        self.right_col_split.setSizes([h // 2, h // 2])
        self._log("Layout restaurado ao padrão (2x2 igual)")

    def _create_layout_slot(self, idx, default_kind="empty", default_channel=0):
        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        frame.setMinimumSize(220, 160)  # não espremer demais
        v = QtWidgets.QVBoxLayout(frame)
        v.setContentsMargins(6, 6, 6, 6); v.setSpacing(4)

        # Toolbar
        tb = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(f"<b>Slot {idx + 1}:</b>")
        lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        tb.addWidget(lbl)
        kind_combo = QtWidgets.QComboBox()
        for label, code in self.PANEL_KINDS:
            kind_combo.addItem(label, code)
        # Define padrão
        idx_def = next((i for i, (_, c) in enumerate(self.PANEL_KINDS) if c == default_kind), 0)
        kind_combo.setCurrentIndex(idx_def)
        tb.addWidget(kind_combo)

        ch_combo = QtWidgets.QComboBox()
        for c in range(MAX_CHANNELS):
            ch_combo.addItem(f"CH{c + 1}")
        ch_combo.setMaximumWidth(80)
        if 0 <= default_channel < MAX_CHANNELS:
            ch_combo.setCurrentIndex(default_channel)
        tb.addWidget(QtWidgets.QLabel("Canal:"))
        tb.addWidget(ch_combo)
        tb.addStretch()
        v.addLayout(tb)

        container = QtWidgets.QWidget()
        c_layout = QtWidgets.QVBoxLayout(container); c_layout.setContentsMargins(0, 0, 0, 0)
        v.addWidget(container, stretch=1)

        slot = {
            "frame": frame, "combo": kind_combo, "ch_combo": ch_combo,
            "container": container, "c_layout": c_layout,
            "widget": None, "kind": "empty",
        }
        self.layout_slots.append(slot)
        kind_combo.currentIndexChanged.connect(
            lambda i, s=slot: self._set_slot_kind(s, self.PANEL_KINDS[i][1]))
        ch_combo.currentIndexChanged.connect(lambda _i, s=slot: self._refresh_slot(s))
        # Aplica tipo padrão
        self._set_slot_kind(slot, default_kind)
        return slot

    def _set_slot_kind(self, slot, kind):
        # Remove widget anterior
        if slot["widget"] is not None:
            slot["c_layout"].removeWidget(slot["widget"])
            slot["widget"].setParent(None)
            slot["widget"].deleteLater()
            slot["widget"] = None
        slot["kind"] = kind
        if kind == "empty":
            return
        w = None
        if kind in ("ts1", "fft", "bands", "spec", "accel"):
            w = pg.PlotWidget(enableMenu=False)
            w.showGrid(x=True, y=True, alpha=0.15)
            w.setMenuEnabled(False)
            if kind == "ts1":
                w.setLabel("left", "uV"); w.setLabel("bottom", "Tempo", units="s")
                slot["curve"] = w.plot(pen=pg.mkPen(COLORS["accent"], width=1.2))
            elif kind == "fft":
                w.setLabel("left", "Amplitude", units="uV")
                w.setLabel("bottom", "Frequência", units="Hz")
                w.setXRange(0, 60)
                slot["curve"] = w.plot(pen=pg.mkPen(COLORS["accent"], width=1.4))
            elif kind == "bands":
                w.setLabel("left", "Potência")
                w.getAxis("bottom").setTicks([list(enumerate(EEG_BANDS.keys()))])
                bars = pg.BarGraphItem(
                    x=list(range(len(EEG_BANDS))),
                    height=[0.0] * len(EEG_BANDS), width=0.6,
                    brush=COLORS["accent"], pen=pg.mkPen(COLORS["accent_dim"]),
                )
                w.addItem(bars); slot["bars"] = bars
                w.setXRange(-0.5, len(EEG_BANDS) - 0.5)
            elif kind == "spec":
                w.setLabel("left", "Freq", units="Hz")
                w.setLabel("bottom", "Tempo", units="s")
                img = pg.ImageItem()
                stops = np.array([0, 0.25, 0.5, 0.75, 1.0])
                colors_m = np.array([
                    [0, 0, 0, 255], [80, 10, 80, 255], [160, 30, 60, 255],
                    [255, 130, 30, 255], [255, 255, 180, 255]], dtype=np.ubyte)
                cmap = pg.ColorMap(stops, colors_m)
                img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
                img.setLevels([-80, 0]); w.addItem(img)
                w.setYRange(0, SPEC_FMAX)
                slot["img"] = img
            elif kind == "accel":
                w.setLabel("left", "g"); w.setLabel("bottom", "Tempo", units="s")
                w.addLegend(offset=(10, 10))
                slot["curves_xyz"] = [
                    w.plot(pen=pg.mkPen(c, width=1.4), name=axis)
                    for c, axis in zip(("#ff5555", "#55ff55", "#5599ff"), ("X", "Y", "Z"))
                ]
        elif kind == "head":
            w = HeadPlotWidget()
            w.set_mapping(self.config.channel_mapping)
            w.set_num_channels(self.num_channels)
        elif kind == "focus":
            w = FocusMeterWidget()
        if w is not None:
            slot["c_layout"].addWidget(w)
            slot["widget"] = w

    def _refresh_slot(self, slot):
        # Apenas re-trigga atualização na próxima chamada
        pass

    def _update_layout_slots(self):
        """Chamado pelos timers para atualizar paineis customizaveis."""
        if not hasattr(self, "layout_slots"):
            return
        data = self._ordered_buffer()
        accel_data = self._ordered_accel()
        n_samples = data.shape[1] if data.size else 0
        if n_samples == 0:
            return
        t_axis = np.arange(n_samples) / SAMPLE_RATE
        band_name = "Alpha"
        if hasattr(self, "topo_band_combo"):
            band_name = self.topo_band_combo.currentText()
        for slot in self.layout_slots:
            k = slot["kind"]
            w = slot["widget"]
            if w is None or k == "empty":
                continue
            ch = slot["ch_combo"].currentIndex()
            if ch >= self.num_channels:
                ch = 0
            try:
                if k == "ts1":
                    slot["curve"].setData(t_axis, data[ch])
                elif k == "fft" and n_samples >= SAMPLE_RATE:
                    freqs, sp = SignalProcessor.compute_fft(data[ch])
                    if freqs.size:
                        slot["curve"].setData(freqs, sp)
                elif k == "bands" and n_samples >= SAMPLE_RATE:
                    powers = SignalProcessor.compute_band_powers(data[ch])
                    slot["bars"].setOpts(height=list(powers.values()))
                elif k == "head" and n_samples >= SAMPLE_RATE:
                    low, high = EEG_BANDS.get(band_name, EEG_BANDS["Alpha"])
                    powers = []
                    for c in range(MAX_CHANNELS):
                        if c < self.num_channels:
                            powers.append(SignalProcessor.compute_band_power(data[c], low, high))
                        else:
                            powers.append(0.0)
                    w.set_powers(powers, band_name)
                elif k == "spec" and n_samples >= SAMPLE_RATE:
                    # Renderiza ultima janela (mais simples que buffer rolante)
                    col = SignalProcessor.compute_psd_column(data[ch])
                    # cria pequena matriz por replicacao
                    arr = np.tile(col.reshape(-1, 1), (1, 30))
                    slot["img"].setImage(arr.T, autoLevels=False)
                    slot["img"].setRect(QtCore.QRectF(0, 0, 7.5, SPEC_FMAX))
                elif k == "accel" and accel_data.shape[1] > 0:
                    t_a = np.arange(accel_data.shape[1]) / SAMPLE_RATE
                    for i in range(3):
                        slot["curves_xyz"][i].setData(t_a, accel_data[i])
                elif k == "focus" and n_samples >= SAMPLE_RATE:
                    focus = SignalProcessor.compute_focus_index(data[ch])
                    w.update_value(focus)
            except Exception:
                pass

    # ==================================================================
    # Teste de Impedância — lead-off detection
    # ==================================================================
    def _refresh_imp_info(self):
        if not hasattr(self, "imp_info_label"): return
        g = self.config.imp_good_max
        a = self.config.imp_acceptable_max
        self.imp_info_label.setText(
            "Avalie a qualidade do contato de cada eletrodo com o couro cabeludo. "
            "O sistema injeta corrente de teste (6 nA pp @ 31,25 Hz) por cada canal "
            "e calcula a impedância a partir da amplitude do sinal (V = I × R).<br><br>"
            f"<b style='color:#22dd33'>● Verde &lt; {g:.1f} kΩ:</b> Ótimo<br>"
            f"<b style='color:{COLORS['warning']}'>● Amarelo {g:.1f}–{a:.1f} kΩ:</b> Aceitável<br>"
            f"<b style='color:{COLORS['error']}'>● Vermelho &gt; {a:.1f} kΩ:</b> Ruim — "
            "re-hidrate o eletrodo, ajuste a touca, verifique o gel/pasta condutiva"
        )

    def _on_imp_thresholds_changed(self, *_):
        try:
            g = float(self.imp_good_spin.value())
            a = float(self.imp_acc_spin.value())
            # garante coerencia: bom <= aceitável
            if g >= a:
                self.imp_acc_spin.blockSignals(True)
                a = g + 1.0
                self.imp_acc_spin.setValue(a)
                self.imp_acc_spin.blockSignals(False)
            self.config.imp_good_max = g
            self.config.imp_acceptable_max = a
            self.config.save()
            self._refresh_imp_info()
            self._log(f"Limites de impedância: verde<{g:.1f} kΩ, amarelo<{a:.1f} kΩ")
        except Exception: pass

    def _start_impedance_test(self):
        if not (self.serial_thread and self.serial_thread.isRunning()
                and self.serial_thread.mode == "hardware"):
            self._log("[Impedância] conecte em modo Hardware para usar este teste", error=True)
            return
        self.imp_test_running = True
        self.imp_start_btn.setEnabled(False)
        self.imp_stop_btn.setEnabled(True)
        # Liga injecao de corrente de teste em cada canal ativo
        # Comando: z<ch><pos><neg>Z, pos=1 = corrente saindo pelo eletrodo positivo
        codes = "12345678QWERTYUI"
        for ch in range(self.num_channels):
            cmd = f"z{codes[ch]}10Z"
            self.serial_thread.send_command(cmd)
            time.sleep(0.05)
        self._log(f"[Impedância] teste iniciado em {self.num_channels} canais "
                  "(aguardando ~3s para FFT estabilizar)...")
        # Timer para calcular impedância a cada 1s
        if not hasattr(self, "imp_timer") or self.imp_timer is None:
            self.imp_timer = QTimer(self)
            self.imp_timer.timeout.connect(self._update_impedance)
        self.imp_timer.start(1000)

    def _stop_impedance_test(self):
        if hasattr(self, "imp_timer") and self.imp_timer is not None:
            self.imp_timer.stop()
        # Desliga injecao de teste em todos os canais
        if self.serial_thread and self.serial_thread.isRunning() \
           and self.serial_thread.mode == "hardware":
            codes = "12345678QWERTYUI"
            for ch in range(self.num_channels):
                cmd = f"z{codes[ch]}00Z"
                self.serial_thread.send_command(cmd)
                time.sleep(0.05)
        self.imp_test_running = False
        self.imp_start_btn.setEnabled(True)
        self.imp_stop_btn.setEnabled(False)
        self._log("[Impedância] teste encerrado, corrente desligada")

    def _update_impedance(self):
        """Computa impedância por canal a partir da amplitude do sinal
        em ~31.25 Hz (frequência do test current do Cyton).
        I = 6 nA pp = 3 nA amplitude => Z(kΩ) = V_uV / 6.
        """
        data = self._ordered_buffer()
        if data.shape[1] < SAMPLE_RATE:
            return
        for ch in range(self.num_channels):
            try:
                ch_data = data[ch, -int(SAMPLE_RATE * 2):]  # ultimos 2s
                freqs, spec = SignalProcessor.compute_fft(ch_data)
                # encontra pico próximo de 31.25 Hz (+- 2 Hz)
                mask = (freqs >= 29.0) & (freqs <= 33.0)
                if not np.any(mask):
                    continue
                amp_uV = float(np.max(spec[mask]))
                imp_kohm = amp_uV / 6.0   # aproximacao
                # Atualiza tabela
                ele_name = self.config.channel_mapping[ch] if ch < len(self.config.channel_mapping) else "—"
                self.imp_table.item(ch, 1).setText(ele_name)
                self.imp_table.item(ch, 2).setText(f"{amp_uV:.2f}")
                imp_item = self.imp_table.item(ch, 3)
                imp_item.setText(f"{imp_kohm:.1f}")
                # Cor + qualidade por faixa (thresholds personalizáveis)
                good_max = float(self.config.imp_good_max)
                acc_max  = float(self.config.imp_acceptable_max)
                qual_item = self.imp_table.item(ch, 4)
                if imp_kohm < good_max:
                    imp_item.setForeground(QtGui.QColor("#22dd33"))
                    if qual_item: qual_item.setText("✓ Ótimo");      qual_item.setForeground(QtGui.QColor("#22dd33"))
                elif imp_kohm < acc_max:
                    imp_item.setForeground(QtGui.QColor(COLORS["warning"]))
                    if qual_item: qual_item.setText("~ Aceitável"); qual_item.setForeground(QtGui.QColor(COLORS["warning"]))
                else:
                    imp_item.setForeground(QtGui.QColor(COLORS["error"]))
                    if qual_item: qual_item.setText("✗ Ruim");       qual_item.setForeground(QtGui.QColor(COLORS["error"]))
            except Exception:
                pass

    # ==================================================================
    # Aba CONFIGURACOES — tema, mapeamento de canais, sessão, snapshots
    # ==================================================================
    # ==================================================================
    # ABA ERP — médias de épocas ao redor de markers
    # ==================================================================
    def _build_erp_tab(self):
        widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(10, 8, 10, 8); outer.setSpacing(6)

        header = QtWidgets.QLabel(
            f"<span style='color:{COLORS['accent']}; font-size:14pt; font-weight:bold;'>"
            "ERP Averager</span>  "
            f"<span style='color:{COLORS['text_dim']};'>"
            "Média de épocas (-200 a +800 ms) ao redor de cada marcador. "
            "Carregue um data.csv com marcadores e clique 'Calcular'."
            "</span>"
        )
        header.setTextFormat(QtCore.Qt.TextFormat.RichText)
        outer.addWidget(header)

        # Controles
        ctrl_row = QtWidgets.QHBoxLayout()
        self.erp_load_btn = QtWidgets.QPushButton("Carregar data.csv...")
        self.erp_load_btn.clicked.connect(self._erp_load_csv)
        ctrl_row.addWidget(self.erp_load_btn)
        ctrl_row.addWidget(QtWidgets.QLabel("Marcador:"))
        self.erp_marker_combo = QtWidgets.QComboBox()
        self.erp_marker_combo.setMinimumWidth(150)
        ctrl_row.addWidget(self.erp_marker_combo)
        ctrl_row.addWidget(QtWidgets.QLabel("Canal:"))
        self.erp_channel_combo = QtWidgets.QComboBox()
        for i in range(MAX_CHANNELS):
            self.erp_channel_combo.addItem(f"CH{i+1}")
        ctrl_row.addWidget(self.erp_channel_combo)
        ctrl_row.addWidget(QtWidgets.QLabel("Pré (ms):"))
        self.erp_pre_spin = QtWidgets.QSpinBox()
        self.erp_pre_spin.setRange(50, 1000); self.erp_pre_spin.setValue(200)
        self.erp_pre_spin.setSuffix(" ms")
        ctrl_row.addWidget(self.erp_pre_spin)
        ctrl_row.addWidget(QtWidgets.QLabel("Pós (ms):"))
        self.erp_post_spin = QtWidgets.QSpinBox()
        self.erp_post_spin.setRange(100, 3000); self.erp_post_spin.setValue(800)
        self.erp_post_spin.setSuffix(" ms")
        ctrl_row.addWidget(self.erp_post_spin)
        self.erp_compute_btn = QtWidgets.QPushButton("Calcular ERP")
        self.erp_compute_btn.clicked.connect(self._erp_compute)
        self.erp_compute_btn.setEnabled(False)
        ctrl_row.addWidget(self.erp_compute_btn)
        ctrl_row.addStretch()
        outer.addLayout(ctrl_row)

        # Plot da forma de onda ERP
        self.erp_plot = pg.PlotWidget(enableMenu=False)
        self.erp_plot.showGrid(x=True, y=True, alpha=0.15)
        self.erp_plot.setLabel("left", "Amplitude", units="µV")
        self.erp_plot.setLabel("bottom", "Tempo após marcador", units="s")
        self.erp_plot.setMenuEnabled(False)
        self.erp_plot.addLegend(offset=(10, 10))
        self.erp_mean_curve = self.erp_plot.plot(
            pen=pg.mkPen(COLORS["accent"], width=2), name="Média (N épocas)"
        )
        self.erp_band_top = self.erp_plot.plot(
            pen=pg.mkPen(COLORS["accent_dim"], width=1,
                          style=QtCore.Qt.PenStyle.DashLine), name="±1 SD"
        )
        self.erp_band_bot = self.erp_plot.plot(
            pen=pg.mkPen(COLORS["accent_dim"], width=1,
                          style=QtCore.Qt.PenStyle.DashLine)
        )
        # Linha vertical em t=0
        zero_line = pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen(COLORS["text_dim"], width=1,
                          style=QtCore.Qt.PenStyle.DashLine)
        )
        self.erp_plot.addItem(zero_line)
        outer.addWidget(self.erp_plot, stretch=1)

        self.erp_info_label = QtWidgets.QLabel("Carregue um CSV para começar.")
        self.erp_info_label.setStyleSheet(f"color: {COLORS['text_dim']};")
        outer.addWidget(self.erp_info_label)

        # Estado
        self._erp_data = None  # dict carregado por _load_session_csv
        return widget

    def _erp_load_csv(self):
        csv_path, _sd = self._pick_session_csv()
        if not csv_path: return
        d = self._load_session_csv(csv_path)
        if not d:
            self.erp_info_label.setText("Falha ao ler CSV"); return
        if not d["markers"]:
            QtWidgets.QMessageBox.information(self, "Sem marcadores",
                "Esse CSV não tem marcadores — não dá para calcular ERP.")
            return
        self._erp_data = d
        # Popula combo de markers
        from collections import Counter
        counts = Counter(lbl for _t, lbl in d["markers"])
        self.erp_marker_combo.clear()
        for lbl, n in counts.most_common():
            self.erp_marker_combo.addItem(f"{lbl}  (n={n})", lbl)
        self.erp_compute_btn.setEnabled(True)
        self.erp_info_label.setText(
            f"Carregado: {len(d['ch_names'])} canais, "
            f"{d['eeg'].shape[1]} amostras @ {d['sr']:.1f} Hz, "
            f"{len(d['markers'])} marcadores ({len(counts)} tipos distintos)"
        )

    def _erp_compute(self):
        if not self._erp_data: return
        d = self._erp_data
        label = self.erp_marker_combo.currentData() or self.erp_marker_combo.currentText().split("  ")[0]
        ch = self.erp_channel_combo.currentIndex()
        if ch >= len(d["ch_names"]): ch = 0
        pre_s = self.erp_pre_spin.value() / 1000.0
        post_s = self.erp_post_spin.value() / 1000.0
        sr = d["sr"]
        n_pre = int(round(pre_s * sr))
        n_post = int(round(post_s * sr))
        x = d["eeg"][ch]
        n_total = x.shape[0]
        # Coleta épocas (em amostras)
        epochs = []
        for t, lbl in d["markers"]:
            if lbl != label: continue
            center = int(round(t * sr))
            i0 = center - n_pre
            i1 = center + n_post
            if i0 < 0 or i1 > n_total: continue
            seg = x[i0:i1].copy()
            # Baseline correction: média dos pré-stimulus
            baseline = float(np.mean(seg[:n_pre]))
            seg -= baseline
            epochs.append(seg)
        if not epochs:
            QtWidgets.QMessageBox.warning(self, "ERP",
                "Nenhuma época válida (verifique se há margem temporal suficiente).")
            return
        ep = np.stack(epochs, axis=0)  # (N, T)
        mean_wave = ep.mean(axis=0)
        sd_wave   = ep.std(axis=0)
        t_axis = np.arange(-n_pre, n_post) / sr
        self.erp_mean_curve.setData(t_axis, mean_wave)
        self.erp_band_top.setData(t_axis, mean_wave + sd_wave)
        self.erp_band_bot.setData(t_axis, mean_wave - sd_wave)
        # Pico positivo + negativo
        idx_pos = int(np.argmax(mean_wave[n_pre:])) + n_pre
        idx_neg = int(np.argmin(mean_wave[n_pre:])) + n_pre
        self.erp_info_label.setText(
            f"ERP de '{label}' em {d['ch_names'][ch]}: "
            f"N={len(epochs)} épocas, "
            f"Pico+ @ {t_axis[idx_pos]*1000:.0f} ms = {mean_wave[idx_pos]:+.2f} µV, "
            f"Pico– @ {t_axis[idx_neg]*1000:.0f} ms = {mean_wave[idx_neg]:+.2f} µV"
        )
        self._audit_event("erp_compute", label=label, channel=d["ch_names"][ch],
                          n_epochs=len(epochs))

    # ==================================================================
    # ABA CONECTIVIDADE — matriz de coerência entre eletrodos
    # ==================================================================
    def _build_connectivity_tab(self):
        widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(widget)
        outer.setContentsMargins(10, 8, 10, 8); outer.setSpacing(6)

        header = QtWidgets.QLabel(
            f"<span style='color:{COLORS['accent']}; font-size:14pt; font-weight:bold;'>"
            "Conectividade Funcional</span>  "
            f"<span style='color:{COLORS['text_dim']};'>"
            "Matriz de coerência (0-1) entre pares de canais na banda escolhida. "
            "Atualiza a cada 2 s sobre os últimos 4 s de sinal."
            "</span>"
        )
        header.setTextFormat(QtCore.Qt.TextFormat.RichText)
        outer.addWidget(header)

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Banda:"))
        self.conn_band_combo = QtWidgets.QComboBox()
        self.conn_band_combo.addItems(list(EEG_BANDS.keys()))
        self.conn_band_combo.setCurrentText("Alpha")
        ctrl.addWidget(self.conn_band_combo)
        ctrl.addStretch()
        self.conn_status = QtWidgets.QLabel("Aguardando dados (precisa de >4s).")
        self.conn_status.setStyleSheet(f"color: {COLORS['text_dim']};")
        ctrl.addWidget(self.conn_status)
        outer.addLayout(ctrl)

        # Imagem da matriz de coerência
        self.conn_plot = pg.PlotWidget(enableMenu=False)
        self.conn_plot.setLabel("left",   "Canal")
        self.conn_plot.setLabel("bottom", "Canal")
        self.conn_plot.setMenuEnabled(False)
        self.conn_image = pg.ImageItem()
        # Colormap (viridis-like)
        stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        colors_v = np.array([
            [ 68,   1,  84, 255],
            [ 59,  82, 139, 255],
            [ 33, 144, 141, 255],
            [ 94, 201,  98, 255],
            [253, 231,  37, 255],
        ], dtype=np.ubyte)
        cmap = pg.ColorMap(stops, colors_v)
        self.conn_image.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        self.conn_image.setLevels([0.0, 1.0])
        self.conn_plot.addItem(self.conn_image)
        outer.addWidget(self.conn_plot, stretch=1)

        # Timer de atualizacao 0.5 Hz
        self._conn_timer = QtCore.QTimer(self)
        self._conn_timer.timeout.connect(self._update_connectivity)
        self._conn_timer.start(2000)
        return widget

    def _update_connectivity(self):
        data = self._ordered_buffer()
        if data.shape[1] < SAMPLE_RATE * 4: return
        last = data[:self.num_channels, -int(SAMPLE_RATE * 4):]
        band = self.conn_band_combo.currentText()
        low, high = EEG_BANDS[band]
        n_ch = last.shape[0]
        # Coerencia magnitude-squared via scipy.signal.coherence
        mat = np.eye(n_ch)
        try:
            for i in range(n_ch):
                for j in range(i + 1, n_ch):
                    f, c = scipy_signal.coherence(
                        last[i], last[j], fs=SAMPLE_RATE,
                        nperseg=min(256, last.shape[1])
                    )
                    mask = (f >= low) & (f < high)
                    if np.any(mask):
                        val = float(np.mean(c[mask]))
                    else:
                        val = 0.0
                    mat[i, j] = mat[j, i] = val
        except Exception as exc:
            self.conn_status.setText(f"Erro: {exc}")
            return
        # Renderiza
        self.conn_image.setImage(mat.T, autoLevels=False)
        self.conn_image.setRect(QtCore.QRectF(0, 0, n_ch, n_ch))
        ticks = [(i + 0.5, f"CH{i+1}") for i in range(n_ch)]
        try:
            self.conn_plot.getAxis("bottom").setTicks([ticks])
            self.conn_plot.getAxis("left").setTicks([ticks])
        except Exception: pass
        # Estatisticas
        off_diag = mat[np.triu_indices(n_ch, k=1)]
        if off_diag.size:
            mn = float(np.mean(off_diag))
            mx = float(np.max(off_diag))
            self.conn_status.setText(
                f"Banda {band} ({low:.1f}–{high:.1f} Hz): "
                f"coerência média {mn:.3f}, máx {mx:.3f}"
            )

    # ==================================================================
    # DEMO MODE 30s — protocolo automatico (eyes open/closed)
    # ==================================================================
    def _start_demo_mode(self):
        if self.serial_thread and self.serial_thread.isRunning():
            QtWidgets.QMessageBox.warning(self, "Demo",
                "Desconecte primeiro para rodar o modo demo.")
            return
        # Configura: simulacao, gravacao automatica, markers a cada 5s
        self.mode_combo.setCurrentIndex(1)  # simulacao
        # Conecta
        self._connect()
        # Aguarda 1s antes de gravar (deixa estabilizar)
        QtCore.QTimer.singleShot(1000, self._demo_step_start_recording)
        self._log("Demo iniciado — duração total ~32s")

    def _demo_step_start_recording(self):
        if not (self.serial_thread and self.serial_thread.isRunning()):
            self._log("Demo abortado (conexão falhou)", error=True); return
        self.subject_edit.setText(f"demo_{datetime.now().strftime('%H%M%S')}")
        self._start_recording()
        # Sequencia de markers
        self._demo_seq = [
            (0,    "olhos_abertos"),
            (5,    "olhos_fechados"),
            (10,   "olhos_abertos"),
            (15,   "olhos_fechados"),
            (20,   "olhos_abertos"),
            (25,   "olhos_fechados"),
        ]
        self._demo_t0 = time.time()
        # Agenda cada marker
        for delay_s, label in self._demo_seq:
            QtCore.QTimer.singleShot(
                int(delay_s * 1000),
                lambda lb=label: self._inject_marker_text(lb)
            )
        # Para a gravacao apos 30s e desconecta
        QtCore.QTimer.singleShot(30_000, self._demo_step_finish)

    def _demo_step_finish(self):
        if self.is_recording:
            self._stop_recording()
        finished_dir = self.current_session_dir
        if self.serial_thread:
            self._disconnect()
        # Tenta gerar PDF automatico da sessão
        if HAS_REPORTLAB and HAS_MPL and finished_dir:
            csv_path = os.path.join(finished_dir, "data.csv")
            if os.path.exists(csv_path):
                d = self._load_session_csv(csv_path)
                if d:
                    try:
                        out_path = os.path.join(finished_dir, "report.pdf")
                        self._generate_pdf_report(d, finished_dir, out_path)
                        self._log(f"PDF do demo gerado: {out_path}")
                        QtWidgets.QMessageBox.information(self, "Demo concluído",
                            f"Demo de 30s finalizada!\n\n"
                            f"Pasta: {finished_dir}\n\n"
                            f"Inclui: data.csv, summary.json (com SHA-256), "
                            f"snapshots/, report.pdf")
                        return
                    except Exception as exc:
                        self._log(f"Falha PDF demo: {exc}", error=True)
        QtWidgets.QMessageBox.information(self, "Demo concluído",
            f"Demo de 30s finalizada!\nPasta: {finished_dir}")

    # ==================================================================
    # EXPORTAÇÃO para formatos científicos
    # ==================================================================
    def _pick_session_csv(self):
        """Diálogo para escolher data.csv de uma sessão. Retorna (csv, dir)."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Escolha o data.csv da sessão",
            self.config.save_directory, "CSV (data.csv);;Todos (*)"
        )
        if not path: return None, None
        return path, os.path.dirname(path)

    @staticmethod
    def _format_sr_check(d):
        """Gera string HTML com o status da detecção de sample rate."""
        sr_d = d.get("sr_declared")
        sr_m = d.get("sr_measured")
        if sr_d and sr_m:
            diff_pct = abs(sr_m - sr_d) / sr_d * 100
            if diff_pct < 0.5:
                col = "#22dd33"; verdict = "✓"
            elif diff_pct < 5.0:
                col = "#ffaa00"; verdict = "!"
            else:
                col = "#ff3355"; verdict = "✗"
            return (f"<span style='color:{col};'>{verdict}</span> "
                    f"<b>SR declarada:</b> {sr_d:.1f} Hz · "
                    f"<b>medida:</b> {sr_m:.2f} Hz · "
                    f"<b>diferença:</b> {diff_pct:.2f}%")
        if sr_m:
            return f"<b>SR medida</b> (sem declaração): {sr_m:.2f} Hz"
        if sr_d:
            return f"<b>SR declarada</b> (sem medição): {sr_d:.1f} Hz"
        return "SR não disponível"

    def _load_session_csv(self, csv_path):
        """Carrega CSV de sessão. Auto-detecta formato:
        - 'native': formato do OpenBiônica (cols *_uV, marker)
        - 'bci_protocol': formato do Data acquisition system.py (Time:125Hz,
           Epoch, <15 canais>, <4 Aux>, Event Id, ..., Class Id, ...)
        Retorna dict {sr, eeg (n_ch, n_samp) µV, ch_names, markers, format,
        trials (BCI), events_csv_path (BCI)}."""
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                header_line = f.readline().strip()
                header = header_line.split(",")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Erro",
                f"Não consegui ler o CSV: {exc}")
            return None

        # ---- Detecção de formato ----
        # BCI: header tem "Time:" + "Event Id" + >= 8 canais do conjunto BCI
        # (não exige mais "Class Id" — variantes antigas não têm)
        has_time = any(h.startswith("Time:") for h in header)
        has_event_id = "Event Id" in header
        bci_chs_found = sum(1 for c in self.BCI_EEG_CHANNELS if c in header)
        is_bci = has_time and has_event_id and bci_chs_found >= 8
        if is_bci:
            return self._load_bci_session_csv(csv_path, header)
        return self._load_native_session_csv(csv_path, header)

    def _load_native_session_csv(self, csv_path, header):
        """Loader para o formato nativo do OpenBiônica."""
        ch_cols = [i for i, h in enumerate(header) if h.endswith("_uV")]
        ch_names = [header[i].replace("_uV", "") for i in ch_cols]
        marker_col = header.index("marker") if "marker" in header else None
        # Defesa: CSVs antigos podem ter header mais largo que as linhas de dados
        # (sessao gravada com num_channels custom > largura real do stream).
        # Mantem apenas colunas que existem de fato na 1a linha -> nao quebra o load.
        try:
            with open(csv_path, "r", encoding="utf-8") as _f:
                next(_f)
                n_fields = len(next(_f).rstrip("\n").split(","))
            if any(i >= n_fields for i in ch_cols):
                keep = [k for k, i in enumerate(ch_cols) if i < n_fields]
                logging.getLogger("eeg").warning(
                    "CSV %s: header com %d canais mas linhas com %d campos; "
                    "carregando %d canais validos.",
                    os.path.basename(csv_path), len(ch_cols), n_fields, len(keep))
                ch_names = [ch_names[k] for k in keep]
                ch_cols  = [ch_cols[k]  for k in keep]
            if marker_col is not None and marker_col >= n_fields:
                marker_col = None
        except StopIteration:
            pass
        except Exception:
            pass
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1,
                          usecols=ch_cols, encoding="utf-8")
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        eeg = data.T
        # SR medida a partir dos timestamps reais (coluna 0)
        sr_measured = None
        try:
            ts = np.loadtxt(csv_path, delimiter=",", skiprows=1,
                            usecols=(0,), encoding="utf-8")
            dt = float(np.median(np.diff(ts)))
            if dt > 0:
                sr_measured = 1.0 / dt
        except Exception: pass
        # Formato nativo não declara SR no header — assume SAMPLE_RATE padrão
        sr_declared = float(SAMPLE_RATE)
        sr = sr_measured if sr_measured else sr_declared
        markers = []
        if marker_col is not None:
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    next(f)
                    for i, line in enumerate(f):
                        parts = line.strip().split(",")
                        if len(parts) > marker_col and parts[marker_col].strip():
                            markers.append((i / sr, parts[marker_col].strip()))
            except Exception: pass
        return {"sr": sr, "sr_declared": sr_declared,
                "sr_measured": sr_measured,
                "n_samples": eeg.shape[1],
                "duration_s": eeg.shape[1] / sr,
                "eeg": eeg, "ch_names": ch_names, "markers": markers,
                "format": "native", "trials": [], "events_csv_path": None}

    # ----- Loader para formato do "Data acquisition system.py" -----
    # Canais EEG fixos do sistema BCI (na ordem do CSV):
    BCI_EEG_CHANNELS = [
        "Cz", "C3", "C4", "Fz", "Pz", "P3", "P4", "F3", "F4",
        "CPz", "FCz", "FP1", "FP2", "T3", "T4",
    ]
    # Correção de rotulagem da montagem: o eletrodo FCz foi gravado pelo
    # "Data acquisition system" sob o NOME DE COLUNA "C2". Mantemos o alias
    # para localizar a coluna no CSV, mas exibimos/posicionamos como FCz.
    BCI_CH_COL_ALIAS = {"FCz": "C2"}
    # IDs de eventos do sistema BCI (do EVENT_IDS)
    BCI_EVENT_NAMES = {
        0:  "baseline",    1: "pre_rest",       2: "cue",
        3:  "mi",          # versões antigas (sem distinção de classe)
        10: "mi_left_hand",11: "mi_right_hand", 12: "mi_dorsi",
        13: "mi_plantar",  20: "iti",           30: "inter_run",
        31: "inter_baseline",
    }
    BCI_CLASS_NAMES = {
        0: "LEFT_HAND",  1: "RIGHT_HAND",
        2: "DORSI",      3: "PLANTAR",
    }

    def _load_bci_session_csv(self, csv_path, header):
        """Loader para CSV gerado pelo Data acquisition system.py
        (formato: Time:125Hz, Epoch, 15 EEG, 4 Aux, Event Id, Event Date,
        Event Duration, Run Idx, Trial Idx, Jitter Type, Class Id, ...)."""
        # Localiza colunas dos canais EEG pelo nome
        ch_cols, ch_names = [], []
        for name in self.BCI_EEG_CHANNELS:
            col = name if name in header else self.BCI_CH_COL_ALIAS.get(name)
            if col is not None and col in header:
                ch_cols.append(header.index(col))
                ch_names.append(name)   # nome de exibição (ex.: coluna "C2" -> rótulo "Fpz")
        if not ch_cols:
            # fallback: assume cols 2..16 (15 canais EEG após Time, Epoch)
            ch_cols = list(range(2, min(17, len(header))))
            ch_names = self.BCI_EEG_CHANNELS[:len(ch_cols)]

        # SR declarada: extraída do header "Time:NNHz"
        sr_declared = 125.0
        for h in header:
            if h.startswith("Time:"):
                try:
                    s = h.split("Time:", 1)[1].rstrip("Hz").strip()
                    sr_declared = float(s)
                except Exception: pass
                break

        try:
            data = np.loadtxt(csv_path, delimiter=",", skiprows=1,
                              usecols=ch_cols, encoding="utf-8")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Erro",
                f"Falha ao carregar dados BCI: {exc}")
            return None
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        eeg = data.T  # shape (n_ch, n_samp)

        # SR medida: tenta primeiro a coluna 0 (Time em segundos); se der valor
        # implausível (~0.5 Hz, 0 Hz, ou >2000 Hz) cai para a coluna "Epoch"
        # (Unix ms) que costuma ser mais confiável em arquivos do Data acquisition.
        sr_measured = None
        def _check_sr(ts, scale_to_s=1.0):
            """ts em unidade qualquer; scale_to_s converte para segundos."""
            if ts.size < 2: return None
            diffs = np.diff(ts) * scale_to_s
            # Remove zeros e negativos (timestamps repetidos)
            valid = diffs[diffs > 0]
            if valid.size < max(10, ts.size // 4):
                return None  # < 25% das amostras com dt válido = não confiável
            dt = float(np.median(valid))
            if dt <= 0: return None
            sr = 1.0 / dt
            # Plausível: 10 Hz - 4000 Hz
            return sr if 10.0 < sr < 4000.0 else None
        # Tenta coluna 0 (Time:NNHz)
        try:
            ts0 = np.loadtxt(csv_path, delimiter=",", skiprows=1,
                              usecols=(0,), encoding="utf-8")
            sr_measured = _check_sr(ts0, scale_to_s=1.0)
        except Exception: pass
        # Fallback: coluna "Epoch" (Unix ms)
        if sr_measured is None and "Epoch" in header:
            try:
                ep_col = header.index("Epoch")
                ep = np.loadtxt(csv_path, delimiter=",", skiprows=1,
                                 usecols=(ep_col,), encoding="utf-8")
                sr_measured = _check_sr(ep, scale_to_s=1e-3)  # ms → s
            except Exception: pass
        # Último fallback: usa duração total (último - primeiro Epoch) / N
        if sr_measured is None and "Epoch" in header:
            try:
                ep_col = header.index("Epoch")
                ep = np.loadtxt(csv_path, delimiter=",", skiprows=1,
                                 usecols=(ep_col,), encoding="utf-8")
                total_s = (float(ep[-1]) - float(ep[0])) / 1000.0
                if total_s > 0:
                    sr_candidate = (len(ep) - 1) / total_s
                    if 10.0 < sr_candidate < 4000.0:
                        sr_measured = sr_candidate
            except Exception: pass

        # SR "oficial" a ser usada no resto do pipeline: prefere a medida
        # quando ela está perto da declarada (±5%); senão usa a declarada
        # (provavelmente erro nos timestamps — ex: arquivo concatenado).
        if sr_measured is not None:
            diff_pct = abs(sr_measured - sr_declared) / sr_declared * 100
            sr = sr_measured if diff_pct < 5.0 else sr_declared
        else:
            sr = sr_declared

        # Detecta unidade: se RMS médio dos canais > 500 µV, provavelmente está
        # em nV (divide por 1000) ou tem offset DC (subtrai média por canal).
        try:
            # Sempre remove offset DC por canal (centra em zero) — corrige drift
            eeg = eeg - np.mean(eeg, axis=1, keepdims=True)
            # Se ainda assim valores típicos > 500 µV, assume escala nV → µV
            rms_mean = float(np.mean([np.sqrt(np.mean(eeg[i]**2))
                                       for i in range(eeg.shape[0])]))
            if rms_mean > 500.0:
                eeg = eeg / 1000.0   # nV → µV (heurística)
                print(f"[BCI loader] valores grandes (RMS={rms_mean:.0f}); "
                      f"assumido nV — convertido para µV")
        except Exception: pass

        # Markers do "Event Id" — registra mudanças de fase
        markers = []
        event_id_col = header.index("Event Id")
        class_id_col = header.index("Class Id") if "Class Id" in header else None
        try:
            usecols = [event_id_col]
            if class_id_col is not None:
                usecols.append(class_id_col)
            arr = np.loadtxt(csv_path, delimiter=",", skiprows=1,
                              usecols=usecols, encoding="utf-8")
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            evs = arr[:, 0].astype(int)
            cls = arr[:, 1].astype(int) if class_id_col is not None else None
            # registra cada mudança de event_id != 0 (entrada na fase)
            prev = -1
            for i, e in enumerate(evs):
                if e != prev:
                    if e != 0:
                        lbl = self.BCI_EVENT_NAMES.get(int(e), f"evt_{e}")
                        # MI: adiciona class name (se Class Id existir)
                        if int(e) in (10, 11, 12, 13) and cls is not None:
                            cn = self.BCI_CLASS_NAMES.get(int(cls[i]), "?")
                            lbl = f"{lbl}_{cn}"
                        markers.append((i / sr, lbl))
                    prev = e
        except Exception: pass

        # Tenta carregar events.csv companheiro (mesmo nome com _events.csv)
        trials = []
        events_csv = None
        # Tenta dois esquemas: <base>_events.csv ou events.csv ao lado
        base_no_ext = csv_path.rsplit(".csv", 1)[0]
        candidates = [base_no_ext + "_events.csv",
                      os.path.join(os.path.dirname(csv_path), "events.csv")]
        for cand in candidates:
            if os.path.isfile(cand):
                events_csv = cand
                break
        if events_csv:
            trials = self._load_bci_events(events_csv)
        else:
            # Sem events.csv companheiro: reconstrói trials a partir do
            # vetor Event Id (cada bloco contíguo de mesmo event_id = 1 trial)
            trials = self._reconstruct_trials_from_event_id(
                csv_path, header, sr)
        return {"sr": sr, "sr_declared": sr_declared,
                "sr_measured": sr_measured,
                "n_samples": eeg.shape[1],
                "duration_s": eeg.shape[1] / sr,
                "eeg": eeg, "ch_names": ch_names, "markers": markers,
                "format": "bci_protocol", "trials": trials,
                "events_csv_path": events_csv}

    def _reconstruct_trials_from_event_id(self, csv_path, header, sr):
        """Quando o events.csv não está disponível, reconstrói os trials lendo
        a coluna 'Event Id' do próprio CSV. Cada bloco contíguo com o mesmo
        event_id != event_id_anterior vira um trial."""
        trials = []
        try:
            event_id_col = header.index("Event Id")
            class_id_col = header.index("Class Id") if "Class Id" in header else None
            usecols = [event_id_col]
            if class_id_col is not None:
                usecols.append(class_id_col)
            arr = np.loadtxt(csv_path, delimiter=",", skiprows=1,
                              usecols=usecols, encoding="utf-8")
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            evs = arr[:, 0].astype(int)
            cls_arr = arr[:, 1].astype(int) if class_id_col is not None else None

            # Mapas reversos para descobrir phase a partir do event_id
            evid_to_phase = {
                0: "baseline", 1: "pre_rest", 2: "cue",
                3: "mi",   # versões antigas (sem class_id explícito)
                10: "mi", 11: "mi", 12: "mi", 13: "mi",
                20: "iti", 30: "inter_run", 31: "inter_baseline",
            }
            evid_to_cls = {10: 0, 11: 1, 12: 2, 13: 3}

            n = len(evs)
            i = 0
            event_idx = 0
            while i < n:
                cur = int(evs[i])
                j = i
                while j < n and int(evs[j]) == cur:
                    j += 1
                # Bloco [i .. j-1] com mesmo event_id cur
                phase = evid_to_phase.get(cur, f"evt_{cur}")
                # class_id: prefere Class Id da coluna, senão deduz do event_id
                if cls_arr is not None and cls_arr[i] >= 0:
                    c = int(cls_arr[i])
                else:
                    c = evid_to_cls.get(cur, -1)
                event_idx += 1
                trials.append({
                    "event_idx":  event_idx,
                    "phase":      phase,
                    "class_id":   c,
                    "class_name": self.BCI_CLASS_NAMES.get(c, "")
                                  if c >= 0 else "",
                    "run_idx":    0,
                    "trial_idx":  event_idx,
                    "start_line": i + 1,        # 1-indexed
                    "end_line":   j,             # inclusive 1-indexed (= j-1+1)
                    "duration":   (j - i) / sr,
                    "foot_side":  "",
                })
                i = j
            return trials
        except Exception as exc:
            print(f"[BCI reconstruct] falha: {exc}")
            return []

    def _load_bci_events(self, events_path):
        """Lê o events.csv do BCI system (15 colunas incluindo phase, class_id,
        csv_start_line, csv_end_line). Retorna lista de trials."""
        trials = []
        try:
            with open(events_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        trials.append({
                            "phase":      row.get("phase", ""),
                            "class_id":   int(row.get("class_id", -1) or -1),
                            "class_name": row.get("class_name", ""),
                            "run_idx":    int(row.get("run_idx", -1) or -1),
                            "trial_idx":  int(row.get("trial_global_idx", -1) or -1),
                            "start_line": int(row.get("csv_start_line", 0) or 0),
                            "end_line":   int(row.get("csv_end_line", 0) or 0),
                            "duration":   float(row.get("actual_duration_s",
                                                       row.get("planned_duration_s", 0)) or 0),
                            "foot_side":  row.get("foot_side", ""),
                        })
                    except Exception: pass
        except Exception as exc:
            print(f"[BCI events] falha lendo {events_path}: {exc}")
        return trials

    def _export_to_edf(self):
        if not HAS_EDF: return
        csv_path, sess_dir = self._pick_session_csv()
        if not csv_path: return
        d = self._load_session_csv(csv_path)
        if not d: return
        out_path = os.path.join(sess_dir, "data.edf")
        w = None
        try:
            import pyedflib  # lazy
            eeg = np.asarray(d["eeg"], dtype=float)
            # Saneia valores nao-finitos (NaN/Inf) — senao o EDF sai corrompido
            if not np.all(np.isfinite(eeg)):
                eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)
            n_ch = eeg.shape[0]
            w = pyedflib.EdfWriter(out_path, n_ch,
                                    file_type=pyedflib.FILETYPE_EDFPLUS)
            ch_info = []
            for i in range(n_ch):
                ch_info.append({
                    "label":        d["ch_names"][i],
                    "dimension":    "uV",
                    "sample_frequency": int(round(d["sr"])),
                    "physical_max": float(np.max(eeg[i]) + 1.0),
                    "physical_min": float(np.min(eeg[i]) - 1.0),
                    "digital_max":  32767,
                    "digital_min": -32768,
                    "transducer":   "EEG electrode",
                    "prefilter":    "",
                })
            w.setSignalHeaders(ch_info)
            # Proveniência (reprodutibilidade): carimba a versão do app no EDF+
            try:
                w.setEquipment(f"{APP_NAME_ASCII} v{APP_VERSION}")  # EDF: ASCII
                w.setRecordingAdditional(CODE_URL[:80])
            except Exception:
                pass
            for t, label in d["markers"]:
                w.writeAnnotation(float(t), -1, str(label))
            w.writeSamples(list(eeg))
            w.close(); w = None
            self._log(f"✓ EDF exportado: {out_path}")
            self._audit_event("export_edf", path=out_path, channels=n_ch)
            QtWidgets.QMessageBox.information(self, "EDF exportado",
                                              f"Arquivo salvo:\n{out_path}")
        except Exception as exc:
            # Fecha o handle ANTES de remover o arquivo / abrir o dialog, senao o
            # .edf parcial fica travado no Windows (WinError 32).
            if w is not None:
                try: w.close()
                except Exception: pass
                w = None
            try: os.remove(out_path)        # nao deixa .edf parcial/invalido
            except OSError: pass
            logging.getLogger("eeg").exception("Falha exportando EDF")
            self._notify_error("E115", str(exc), exc=exc)

    def _export_to_fif(self):
        if not HAS_MNE: return
        csv_path, sess_dir = self._pick_session_csv()
        if not csv_path: return
        d = self._load_session_csv(csv_path)
        if not d: return
        out_path = os.path.join(sess_dir, "data_raw.fif")
        try:
            import mne  # lazy
            data_V = d["eeg"] * 1e-6
            info = mne.create_info(
                ch_names=d["ch_names"], sfreq=d["sr"],
                ch_types=["eeg"] * len(d["ch_names"]),
            )
            raw = mne.io.RawArray(data_V, info, verbose=False)
            try:
                raw.info["description"] = f"{APP_NAME} v{APP_VERSION} | {CODE_URL}"
            except Exception:
                pass
            if d["markers"]:
                onsets = [t for t, _ in d["markers"]]
                durs   = [0.0] * len(onsets)
                descs  = [str(lbl) for _, lbl in d["markers"]]
                raw.set_annotations(mne.Annotations(onsets, durs, descs))
            raw.save(out_path, overwrite=True, verbose=False)
            # Gera script de análise pronto-para-usar
            script_path = os.path.join(sess_dir, "analyze_mne.py")
            self._write_mne_analysis_script(script_path, os.path.basename(out_path),
                                             d["ch_names"], d["sr"])
            self._log(f"FIF exportado: {out_path} + script analyze_mne.py")
            self._audit_event("export_fif", path=out_path,
                              channels=len(d["ch_names"]))
            QtWidgets.QMessageBox.information(
                self, "FIF + Script MNE exportados",
                f"<b>Arquivos gerados em:</b><br>{sess_dir}<br><br>"
                f"<code>{os.path.basename(out_path)}</code> — dados em formato FIF<br>"
                f"<code>analyze_mne.py</code> — script Python pronto com:<br>"
                "&nbsp;&nbsp;• Bandpass 1-40 Hz<br>"
                "&nbsp;&nbsp;• ICA para limpeza de artefatos<br>"
                "&nbsp;&nbsp;• PSD por banda (δ θ α β γ)<br>"
                "&nbsp;&nbsp;• Time-frequency (Morlet wavelet)<br>"
                "&nbsp;&nbsp;• Eventos extraídos dos markers<br><br>"
                "Execute: <code>python analyze_mne.py</code>"
            )
        except Exception as exc:
            self._notify_error("E116", str(exc), exc=exc)

    @staticmethod
    def _write_mne_analysis_script(out_py, fif_name, ch_names, sr):
        """Gera um script Python pronto para análise EEGLAB-class com MNE."""
        content = f'''"""
Análise EEG com MNE-Python — gerado automaticamente pelo OpenBiônica.

Arquivo FIF: {fif_name}
Canais: {len(ch_names)}  ({", ".join(ch_names[:8])}{", ..." if len(ch_names) > 8 else ""})
SR: {sr} Hz

Pré-requisitos:
    pip install mne matplotlib scikit-learn

Execute:
    python {os.path.basename(out_py)}
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import mne

# ============================================================
# 1) CARREGAR DADOS
# ============================================================
fif_path = os.path.join(os.path.dirname(__file__), "{fif_name}")
raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
print(raw.info)

# ============================================================
# 2) PRÉ-PROCESSAMENTO BÁSICO
# ============================================================
raw.filter(l_freq=1.0, h_freq=40.0, fir_design="firwin", verbose=False)
raw.notch_filter(freqs=[50.0, 60.0], verbose=False)

# Re-referenciação CAR
raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)

# ============================================================
# 3) ICA — REMOÇÃO DE ARTEFATOS (BLINK / MUSCLE)
# ============================================================
ica = mne.preprocessing.ICA(n_components=min(15, len(raw.ch_names)),
                             random_state=42, max_iter="auto",
                             method="fastica", verbose=False)
ica.fit(raw)
# Auto-detectar componentes de blink em Fp1/Fp2 (se existir)
eog_ch = [c for c in ("Fp1", "Fp2") if c in raw.ch_names]
if eog_ch:
    try:
        eog_inds, _ = ica.find_bads_eog(raw, ch_name=eog_ch[0], verbose=False)
        ica.exclude = eog_inds
        print(f"ICA: {{len(eog_inds)}} componentes EOG detectados e excluídos.")
    except Exception:
        pass
raw_clean = ica.apply(raw.copy(), verbose=False)

# ============================================================
# 4) POTÊNCIA POR BANDA (Welch PSD)
# ============================================================
psd = raw_clean.compute_psd(method="welch", fmin=0.5, fmax=50, verbose=False)
freqs = psd.freqs
power = psd.get_data()  # (n_channels, n_freqs)

bands = {{"Delta":(0.5,4),"Theta":(4,8),"Alpha":(8,13),
          "Beta":(13,30),"Gamma":(30,50)}}
band_power = {{}}
for name, (lo, hi) in bands.items():
    mask = (freqs >= lo) & (freqs < hi)
    band_power[name] = power[:, mask].mean(axis=1)

print("\\n=== POTÊNCIA MÉDIA POR BANDA (µV²/Hz) ===")
for name, vals in band_power.items():
    print(f"  {{name:5s}}: {{vals.mean():.3e}}")

# ============================================================
# 5) TIME-FREQUENCY (Morlet wavelet) — primeiro canal
# ============================================================
events = mne.events_from_annotations(raw_clean, verbose=False)[0]
if len(events) > 0:
    epochs = mne.Epochs(raw_clean, events, tmin=-1.0, tmax=2.0,
                         baseline=(None, 0), preload=True, verbose=False)
    freqs_tf = np.arange(4, 40, 2)
    power_tf = mne.time_frequency.tfr_morlet(
        epochs, freqs=freqs_tf, n_cycles=freqs_tf / 2,
        return_itc=False, decim=3, verbose=False)
    fig = power_tf.plot([0], baseline=(-0.5, 0), mode="logratio",
                         title="Time-Frequency (canal 0)", show=False)
    fig.savefig(os.path.join(os.path.dirname(__file__),
                              "tf_channel0.png"), dpi=120)
    print("\\nTime-frequency salvo: tf_channel0.png")

# ============================================================
# 6) RELATÓRIO MNE.REPORT
# ============================================================
report = mne.Report(title="Análise EEG — {fif_name}", verbose=False)
report.add_raw(raw=raw_clean, title="Sinal limpo (CAR + bandpass + ICA)")
report.add_ica(ica=ica, inst=raw, title="ICA")
report.save(os.path.join(os.path.dirname(__file__),
                          "report_mne.html"), overwrite=True,
             open_browser=False, verbose=False)
print("\\nRelatório HTML: report_mne.html")
print("Análise completa.")
'''
        with open(out_py, "w", encoding="utf-8") as f:
            f.write(content)

    def _export_to_bids(self):
        """Exporta sessão no formato BIDS-EEG.

        Estrutura:
            <bids_root>/
              dataset_description.json
              participants.tsv
              sub-XX/
                ses-YY/
                  eeg/
                    sub-XX_ses-YY_task-rest_eeg.csv  (data)
                    sub-XX_ses-YY_task-rest_eeg.json (sidecar)
                    sub-XX_ses-YY_task-rest_channels.tsv
                    sub-XX_ses-YY_task-rest_events.tsv
        """
        csv_path, sess_dir = self._pick_session_csv()
        if not csv_path: return
        d = self._load_session_csv(csv_path)
        if not d: return
        # Pergunta pasta-raiz BIDS
        bids_root = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Escolher pasta raiz BIDS (será criada se vazia)",
            os.path.dirname(sess_dir)
        )
        if not bids_root: return
        try:
            # Identificação
            v = d.get("volunteer") or {}
            vid = (v.get("vid") if isinstance(v, dict) else "") or "01"
            sub_id = "sub-" + re.sub(r"[^A-Za-z0-9]", "", str(vid))[:8] or "sub-01"
            # ses- = timestamp da sessão
            sess_basename = os.path.basename(sess_dir)
            ses_id = "ses-" + re.sub(r"[^A-Za-z0-9]", "", sess_basename)[:16]
            task = "rest"
            eeg_dir = os.path.join(bids_root, sub_id, ses_id, "eeg")
            os.makedirs(eeg_dir, exist_ok=True)

            base = f"{sub_id}_{ses_id}_task-{task}"
            data_path     = os.path.join(eeg_dir, f"{base}_eeg.csv")
            sidecar_path  = os.path.join(eeg_dir, f"{base}_eeg.json")
            channels_path = os.path.join(eeg_dir, f"{base}_channels.tsv")
            events_path   = os.path.join(eeg_dir, f"{base}_events.tsv")

            # 1) Copia data.csv como arquivo de dados (mantemos formato CSV
            #    para compatibilidade — BIDS aceita CSV/EDF/BDF)
            import shutil
            shutil.copy2(csv_path, data_path)

            # 2) Sidecar JSON
            sidecar = {
                "TaskName":               task,
                "SamplingFrequency":      float(d.get("sr", SAMPLE_RATE)),
                "EEGReference":           "TBD (re-referenciação aplicada em runtime)",
                "PowerLineFrequency":     float(self.notch_freq.currentText())
                                          if hasattr(self, "notch_freq") else 60.0,
                "SoftwareFilters":        {
                    "HighPass":  float(self.bp_low.value())  if hasattr(self, "bp_low")  else "n/a",
                    "LowPass":   float(self.bp_high.value()) if hasattr(self, "bp_high") else "n/a",
                    "Notch":     bool(self.notch_enable.isChecked()) if hasattr(self, "notch_enable") else False,
                },
                "EEGChannelCount":        len(d.get("ch_names", [])),
                "EEGGround":              "n/a",
                "Manufacturer":           "OpenBionica",
                "ManufacturersModelName": APP_NAME_ASCII + " v" + APP_VERSION,
                "RecordingDuration":      float(d.get("duration", 0.0)),
                "RecordingType":          "continuous",
            }
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump(sidecar, f, ensure_ascii=False, indent=2)

            # 3) Channels TSV
            with open(channels_path, "w", encoding="utf-8") as f:
                f.write("name\ttype\tunits\tsampling_frequency\n")
                for ch_name in d.get("ch_names", []):
                    f.write(f"{ch_name}\tEEG\tuV\t{d.get('sr', SAMPLE_RATE)}\n")

            # 4) Events TSV
            with open(events_path, "w", encoding="utf-8") as f:
                f.write("onset\tduration\ttrial_type\n")
                for t, label in d.get("markers", []):
                    f.write(f"{t:.4f}\t0\t{label}\n")

            # 5) dataset_description.json (raiz BIDS)
            ds_desc = os.path.join(bids_root, "dataset_description.json")
            if not os.path.exists(ds_desc):
                with open(ds_desc, "w", encoding="utf-8") as f:
                    json.dump({
                        "Name":           f"{APP_NAME_ASCII} dataset",
                        "BIDSVersion":    "1.8.0",
                        "DatasetType":    "raw",
                        "Authors":        [APP_AUTHORS],
                        "GeneratedBy":    [{"Name": APP_NAME_ASCII, "Version": APP_VERSION,
                                            "CodeURL": CODE_URL}],
                    }, f, ensure_ascii=False, indent=2)

            # 6) participants.tsv (anexa se não existir)
            participants_path = os.path.join(bids_root, "participants.tsv")
            existing = ""
            if os.path.exists(participants_path):
                with open(participants_path, "r", encoding="utf-8") as f:
                    existing = f.read()
            with open(participants_path, "w", encoding="utf-8") as f:
                if not existing:
                    f.write("participant_id\tage\tsex\tgroup\n")
                else:
                    f.write(existing)
                # Adiciona linha se não estiver presente
                if sub_id not in existing:
                    age = str(v.get("idade", "n/a"))   if isinstance(v, dict) else "n/a"
                    sex = str(v.get("sexo",  "n/a"))[:1] if isinstance(v, dict) else "n/a"
                    f.write(f"{sub_id}\t{age}\t{sex}\tcontrol\n")

            self._log(f"Exportação BIDS concluída: {sub_id}/{ses_id}")
            self._audit_event("export_bids", path=eeg_dir,
                              subject=sub_id, session=ses_id)
            QtWidgets.QMessageBox.information(
                self, "BIDS-EEG exportado",
                f"<b>Estrutura BIDS criada em:</b><br>{bids_root}<br><br>"
                f"<b>Sujeito:</b> {sub_id}<br>"
                f"<b>Sessão:</b> {ses_id}<br><br>"
                "Pronto para uso com EEGLAB / MNE-BIDS / Brainstorm / FieldTrip."
            )
        except Exception as exc:
            self._notify_error("E117", str(exc), exc=exc)

    def _export_pdf_report(self):
        if not (HAS_REPORTLAB and HAS_MPL): return
        csv_path, sess_dir = self._pick_session_csv()
        if not csv_path: return
        d = self._load_session_csv(csv_path)
        if not d: return
        out_path = os.path.join(sess_dir, "report.pdf")
        try:
            self._generate_pdf_report(d, sess_dir, out_path)
            self._log(f"✓ PDF gerado: {out_path}")
            self._audit_event("export_pdf", path=out_path)
            QtWidgets.QMessageBox.information(self, "PDF gerado",
                                              f"Arquivo salvo:\n{out_path}")
        except Exception as exc:
            self._notify_error("E118", str(exc), exc=exc)

    def _generate_pdf_report(self, d, sess_dir, out_path):
        """PDF de uma página com 3 figuras (timeseries, FFT, banda × canal)."""
        # lazy: matplotlib e reportlab só carregam ao gerar o PDF
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.units import cm
        tmp_dir = os.path.join(sess_dir, "_pdf_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        eeg, sr, names = d["eeg"], d["sr"], d["ch_names"]
        n_ch, n_samp = eeg.shape
        t_axis = np.arange(n_samp) / sr

        n_show = min(n_samp, int(10 * sr))
        fig, axes = plt.subplots(n_ch, 1, figsize=(8, 0.9 * n_ch), sharex=True)
        if n_ch == 1: axes = [axes]
        for i in range(n_ch):
            axes[i].plot(t_axis[:n_show], eeg[i, :n_show], lw=0.6, color="#5a8a00")
            axes[i].set_ylabel(names[i], fontsize=8)
            axes[i].grid(True, alpha=0.3)
        axes[-1].set_xlabel("Tempo (s)")
        fig.suptitle("Sinal EEG — primeiros 10 segundos", fontsize=11)
        fig.tight_layout()
        ts_path = os.path.join(tmp_dir, "timeseries.png")
        fig.savefig(ts_path, dpi=110); plt.close(fig)

        n_fft = min(n_samp, int(sr * 4))
        avg_spec = np.zeros(n_fft // 2 + 1)
        for i in range(n_ch):
            win = np.hanning(n_fft)
            spec = np.abs(rfft(eeg[i, -n_fft:] * win)) * 2.0 / n_fft
            avg_spec += spec
        avg_spec /= n_ch
        freqs = rfftfreq(n_fft, 1.0 / sr)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.semilogy(freqs, avg_spec, color="#5a8a00")
        ax.set_xlim(0, 60); ax.set_xlabel("Frequência (Hz)")
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title("Espectro médio (FFT) — todos os canais")
        for _band, (lo, hi) in EEG_BANDS.items():
            ax.axvspan(lo, hi, alpha=0.1)
        ax.grid(True, alpha=0.3)
        fft_path = os.path.join(tmp_dir, "fft.png")
        fig.tight_layout(); fig.savefig(fft_path, dpi=110); plt.close(fig)

        bands_data = np.zeros((n_ch, len(EEG_BANDS)))
        for i in range(n_ch):
            f, psd = scipy_signal.welch(eeg[i], fs=sr, nperseg=min(256, n_samp))
            for j, (lo, hi) in enumerate(EEG_BANDS.values()):
                mask = (f >= lo) & (f < hi)
                bands_data[i, j] = (float(_TRAPEZOID(psd[mask], f[mask]))
                                    if np.any(mask) else 0.0)
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(bands_data, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(EEG_BANDS))); ax.set_xticklabels(list(EEG_BANDS.keys()))
        ax.set_yticks(range(n_ch)); ax.set_yticklabels(names, fontsize=8)
        ax.set_title("Potência por canal × banda")
        fig.colorbar(im, ax=ax, label="µV²/Hz")
        bands_path = os.path.join(tmp_dir, "bands.png")
        fig.tight_layout(); fig.savefig(bands_path, dpi=110); plt.close(fig)

        # ---- Avaliação de qualidade (QC) por canal -> veredito ----
        qc_ok = qc_noisy = qc_bad = 0
        for i in range(n_ch):
            x = eeg[i]
            rms = float(np.sqrt(np.mean(x * x)))
            pp = float(np.max(x) - np.min(x))
            try:
                fq, psd = scipy_signal.welch(x, fs=sr, nperseg=min(256, n_samp))
                lr = float(np.sum(psd[(fq >= 55) & (fq <= 65)])) / max(
                    float(np.sum(psd[(fq >= 1) & (fq <= 80)])), 1e-12)
            except Exception:
                lr = 0.0
            if pp > 2000 or rms > 500 or rms < 0.5:
                qc_bad += 1
            elif lr > 0.40 or rms > 150:
                qc_noisy += 1
            else:
                qc_ok += 1
        if qc_bad == 0 and qc_noisy <= max(1, n_ch // 4):
            qc_verdict, qc_color = "APTO", (0.11, 0.62, 0.46)
        elif qc_bad <= max(1, n_ch // 4):
            qc_verdict, qc_color = "DUVIDOSO", (0.84, 0.54, 0.0)
        else:
            qc_verdict, qc_color = "DESCARTAR", (0.83, 0.21, 0.31)

        c = rl_canvas.Canvas(out_path, pagesize=A4)
        W, H = A4
        c.setFont("Helvetica-Bold", 16)
        c.drawString(2 * cm, H - 2 * cm, "Relatorio de Sessao EEG")
        c.setFont("Helvetica", 10)
        c.drawString(2 * cm, H - 2.7 * cm,
                     f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        c.drawString(2 * cm, H - 3.2 * cm,
                     f"Sujeito: {self.config.subject}  |  Canais: {n_ch}  |  "
                     f"Fs: {sr:.1f} Hz  |  Duracao: {n_samp/sr:.1f} s  |  "
                     f"Marcadores: {len(d['markers'])}")
        # Veredito de qualidade (QC)
        c.setFillColorRGB(*qc_color)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(2 * cm, H - 3.9 * cm,
                     f"Qualidade do sinal: {qc_verdict}  "
                     f"({qc_ok} OK, {qc_noisy} ruidosos, {qc_bad} ruins de {n_ch} canais)")
        c.setFillColorRGB(0, 0, 0)
        c.drawImage(ts_path,    2*cm, H - 12*cm, width=17*cm, height=8*cm,
                    preserveAspectRatio=True, anchor="c")
        c.drawImage(fft_path,   2*cm, H - 19*cm, width=8.5*cm, height=6*cm,
                    preserveAspectRatio=True, anchor="c")
        c.drawImage(bands_path, 10.5*cm, H - 19*cm, width=8.5*cm, height=6*cm,
                    preserveAspectRatio=True, anchor="c")
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(2 * cm, 1 * cm,
                     f"Gerado por {APP_NAME} v{APP_VERSION} / {APP_AUTHORS} "
                     f"- {CODE_URL}")
        c.save()
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception: pass

    def _build_settings_tab(self):
        """Aba Configurações — usa sub-abas para evitar rolagem na vertical."""
        outer = QtWidgets.QWidget()
        outer_layout = QtWidgets.QVBoxLayout(outer)
        outer_layout.setContentsMargins(6, 6, 6, 6)
        # Sub-tabs internas
        sub_tabs = QtWidgets.QTabWidget()
        sub_tabs.setDocumentMode(True)
        outer_layout.addWidget(sub_tabs)

        # ============== Sub-aba: TEMA & CORES ==============
        theme_page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(theme_page)
        layout.setContentsMargins(10, 8, 10, 8); layout.setSpacing(8)

        # ---- Tema ----
        theme_group = QtWidgets.QGroupBox("Tema (paleta de cores)")
        tl = QtWidgets.QHBoxLayout(theme_group)
        tl.addWidget(QtWidgets.QLabel("Selecione:"))
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(list(THEMES.keys()))
        self.theme_combo.setCurrentText(self.config.theme)
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        tl.addWidget(self.theme_combo)
        tl.addStretch()
        tl.addWidget(QtWidgets.QLabel(
            "(troca instantânea — salvo automaticamente em config.json)"))
        layout.addWidget(theme_group)

        # ---- Idioma (i18n) ----
        lang_group = QtWidgets.QGroupBox("Idioma / Language / Idioma")
        lgl = QtWidgets.QHBoxLayout(lang_group)
        lgl.addWidget(QtWidgets.QLabel("Selecione:"))
        self.lang_combo = QtWidgets.QComboBox()
        # Itens com label visível + dado interno (código)
        for code, label in I18N.LANGUAGES.items():
            self.lang_combo.addItem(label, userData=code)
        # Seleciona o idioma atual
        cur_code = getattr(self.config, "language", "pt")
        idx = self.lang_combo.findData(cur_code)
        if idx >= 0:
            self.lang_combo.setCurrentIndex(idx)
        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        lgl.addWidget(self.lang_combo)
        lgl.addStretch()
        lang_hint = QtWidgets.QLabel(
            "Reinicie o app após trocar de idioma para aplicar em todas as telas."
        )
        lang_hint.setStyleSheet(f"color: {COLORS['text_dim']}; font-style: italic;")
        lgl.addWidget(lang_hint)
        layout.addWidget(lang_group)

        # ---- Editor de tema personalizado ----
        custom_group = QtWidgets.QGroupBox("Editor de Tema Personalizado")
        cgl = QtWidgets.QVBoxLayout(custom_group)
        cgl_info = QtWidgets.QLabel(
            "Clique em qualquer cor abaixo para abrir o seletor (pode arrastar HSV "
            "ou colar codigo hex). Salve como tema personalizado para reutilizar depois."
        )
        cgl_info.setStyleSheet(f"color: {COLORS['text_dim']};")
        cgl_info.setWordWrap(True); cgl.addWidget(cgl_info)

        roles = [
            ("background",  "Fundo principal"),
            ("surface",     "Superfície (cards/header)"),
            ("surface_alt", "Superfície alt. (inputs)"),
            ("table_bg",    "Fundo das tabelas"),
            ("table_alt",   "Linhas alternadas tabela"),
            ("border",      "Borda"),
            ("text",        "Texto principal"),
            ("text_dim",    "Texto secundario"),
            ("accent",      "Cor de destaque"),
            ("accent_dim",  "Destaque escurecido"),
            ("error",       "Erro / desconectado"),
            ("warning",     "Aviso"),
            ("expansion",   "Destaque expansão"),
        ]
        self.color_buttons = {}
        self.color_edits   = {}
        cg = QtWidgets.QGridLayout()
        for i, (key, label) in enumerate(roles):
            row, col = i // 2, (i % 2) * 4
            lbl = QtWidgets.QLabel(label + ":")
            cg.addWidget(lbl, row, col)
            btn = QtWidgets.QPushButton()
            btn.setFixedWidth(34); btn.setFixedHeight(24)
            btn.setStyleSheet(
                f"background-color: {COLORS[key]}; border: 1px solid {COLORS['border']};")
            btn.clicked.connect(lambda _ck, k=key: self._pick_color(k))
            cg.addWidget(btn, row, col + 1)
            edit = QtWidgets.QLineEdit(COLORS[key])
            edit.setFixedWidth(86); edit.setPlaceholderText("#RRGGBB")
            edit.editingFinished.connect(lambda k=key: self._hex_color_changed(k))
            cg.addWidget(edit, row, col + 2)
            self.color_buttons[key] = btn
            self.color_edits[key]   = edit
        cgl.addLayout(cg)

        btn_row2 = QtWidgets.QHBoxLayout()
        apply_custom = QtWidgets.QPushButton("Aplicar agora")
        apply_custom.clicked.connect(self._apply_custom_colors)
        btn_row2.addWidget(apply_custom)
        save_custom = QtWidgets.QPushButton("Salvar como novo tema")
        save_custom.clicked.connect(self._save_custom_theme)
        btn_row2.addWidget(save_custom)
        self.delete_theme_btn = QtWidgets.QPushButton("Deletar tema")
        self.delete_theme_btn.clicked.connect(self._delete_current_custom_theme)
        btn_row2.addWidget(self.delete_theme_btn)
        reset_btn = QtWidgets.QPushButton("Reverter para o tema selecionado")
        reset_btn.clicked.connect(self._reload_color_pickers_from_active)
        btn_row2.addWidget(reset_btn)
        btn_row2.addStretch()
        cgl.addLayout(btn_row2)
        # Atualiza o botão Deletar conforme tema atual (custom ou não)
        self._refresh_delete_theme_button()
        layout.addWidget(custom_group)
        layout.addStretch()
        sub_tabs.addTab(theme_page, "Tema e Cores")

        # ============== Sub-aba: CANAIS (mapeamento 10-20) ==============
        ch_page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(ch_page)
        layout.setContentsMargins(10, 8, 10, 8); layout.setSpacing(8)

        # ---- Mapeamento de canais ----
        map_group = QtWidgets.QGroupBox(
            "Mapeamento de Canais (qual eletrodo cada CHn representa)"
        )
        ml = QtWidgets.QVBoxLayout(map_group)
        info = QtWidgets.QLabel(
            "Selecione manualmente qual eletrodo do sistema 10-20 cada canal "
            "ocupa. Inclui Fz, Cz, Pz, FCz, CPz, POz, Oz, etc. A configuração "
            "fica salva e e usada no Head Plot/Heatmap, FFT e gravação CSV."
        )
        info.setStyleSheet(f"color: {COLORS['text_dim']};")
        info.setWordWrap(True)
        ml.addWidget(info)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(8); grid.setVerticalSpacing(4)
        self.map_combos = []
        # Layout 8 linhas x 2 canais por linha (4 colunas: lbl1, cb1, lbl2, cb2).
        # Cabe todos os 16 canais SEM corte horizontal mesmo em telas pequenas.
        grid.setHorizontalSpacing(10); grid.setVerticalSpacing(4)
        for ch in range(MAX_CHANNELS):
            lbl = QtWidgets.QLabel(f"CH{ch + 1}")
            lbl.setStyleSheet(f"color: {CHANNEL_COLORS[ch]}; font-weight: bold;")
            lbl.setMinimumWidth(46)
            cb = QtWidgets.QComboBox()
            cb.addItems(ELECTRODE_NAMES)
            cb.setMinimumWidth(110)
            cur = self.config.channel_mapping[ch] if ch < len(self.config.channel_mapping) else DEFAULT_MAPPING[ch]
            idx = cb.findText(cur)
            cb.setCurrentIndex(idx if idx >= 0 else 0)
            cb.currentTextChanged.connect(self._on_mapping_changed)
            self.map_combos.append(cb)
            row, col = ch // 2, (ch % 2) * 2
            grid.addWidget(lbl, row, col)
            grid.addWidget(cb,  row, col + 1)
        ml.addLayout(grid)

        btn_row = QtWidgets.QHBoxLayout()
        save_map = QtWidgets.QPushButton("Salvar mapeamento")
        save_map.clicked.connect(self._save_mapping_and_apply)
        reset_map = QtWidgets.QPushButton("Restaurar mapeamento padrão")
        reset_map.clicked.connect(self._reset_mapping_to_default)
        btn_row.addWidget(save_map); btn_row.addWidget(reset_map); btn_row.addStretch()
        ml.addLayout(btn_row)
        layout.addWidget(map_group)
        layout.addStretch()
        sub_tabs.addTab(ch_page, "Mapeamento de Canais")

        # ============== Sub-aba: SESSÃO ==============
        sess_page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(sess_page)
        layout.setContentsMargins(10, 8, 10, 8); layout.setSpacing(8)

        # ---- Sessão ----
        sess_group = QtWidgets.QGroupBox("Organização de Sessões")
        sl = QtWidgets.QGridLayout(sess_group)
        sl.addWidget(QtWidgets.QLabel("Nome do sujeito/participante:"), 0, 0)
        self.subject_edit = QtWidgets.QLineEdit(self.config.subject)
        self.subject_edit.textChanged.connect(self._on_subject_changed)
        sl.addWidget(self.subject_edit, 0, 1, 1, 2)

        sl.addWidget(QtWidgets.QLabel("Template do nome da pasta:"), 1, 0)
        self.template_edit = QtWidgets.QLineEdit(self.config.session_template)
        self.template_edit.textChanged.connect(self._on_template_changed)
        sl.addWidget(self.template_edit, 1, 1, 1, 2)

        sl.addWidget(QtWidgets.QLabel("Variáveis disponiveis:"), 2, 0)
        vars_lbl = QtWidgets.QLabel(
            "{subject}  {date}  {time}  {datetime}  {channels}  — ex: "
            "EEG_{subject}_{datetime}"
        )
        vars_lbl.setStyleSheet(f"color: {COLORS['text_dim']}; font-style: italic;")
        sl.addWidget(vars_lbl, 2, 1, 1, 2)

        sl.addWidget(QtWidgets.QLabel("Snapshot a cada (s):"), 3, 0)
        self.snapshot_spin = QtWidgets.QSpinBox()
        self.snapshot_spin.setRange(0, 600); self.snapshot_spin.setSuffix(" s")
        self.snapshot_spin.setValue(self.config.snapshot_interval)
        self.snapshot_spin.valueChanged.connect(self._on_snapshot_changed)
        sl.addWidget(self.snapshot_spin, 3, 1)
        sl.addWidget(QtWidgets.QLabel("(0 = desativa snapshots periódicos)"), 3, 2)

        sl.addWidget(QtWidgets.QLabel("Previa do nome da próxima sessão:"), 4, 0)
        self.session_preview = QtWidgets.QLabel("")
        self.session_preview.setStyleSheet(
            f"color: {COLORS['accent']}; font-family: {FONT_DATA_STACK};")
        self._refresh_session_preview()
        sl.addWidget(self.session_preview, 4, 1, 1, 2)

        save_btn = QtWidgets.QPushButton("Salvar configurações")
        save_btn.clicked.connect(self._save_all_settings)
        sl.addWidget(save_btn, 5, 0, 1, 3)
        layout.addWidget(sess_group)

        # ---- Pasta de salvamento (configuravel) ----
        savedir_group = QtWidgets.QGroupBox("Pasta de Salvamento das Sessões")
        sdl = QtWidgets.QVBoxLayout(savedir_group)
        sdl_info = QtWidgets.QLabel(
            "Onde os CSVs, logs e snapshots de cada sessão serao gravados. "
            f"O padrão e <code>sessions/</code> ao lado do .py "
            "(portátil — mesmo se mover o script de pasta)."
        )
        sdl_info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        sdl_info.setWordWrap(True)
        sdl_info.setStyleSheet(f"color: {COLORS['text_dim']};")
        sdl.addWidget(sdl_info)
        sd_row = QtWidgets.QHBoxLayout()
        self.savedir_label = QtWidgets.QLineEdit(self.config.save_directory)
        self.savedir_label.setReadOnly(False)   # editavel via texto também
        self.savedir_label.editingFinished.connect(self._on_savedir_text_changed)
        sd_row.addWidget(self.savedir_label, stretch=1)
        sd_browse = QtWidgets.QPushButton("Procurar...")
        sd_browse.clicked.connect(self._browse_save_directory)
        sd_row.addWidget(sd_browse)
        sd_default = QtWidgets.QPushButton("Padrão (junto ao .py)")
        sd_default.clicked.connect(self._reset_save_directory_to_default)
        sd_row.addWidget(sd_default)
        sdl.addLayout(sd_row)
        layout.addWidget(savedir_group)

        # ---- Exportação para formatos científicos ----
        export_group = QtWidgets.QGroupBox("Exportar Sessão (escolher .csv ou pasta de sessão)")
        eg = QtWidgets.QGridLayout(export_group)
        eg.addWidget(QtWidgets.QLabel("Formato:"), 0, 0)
        self.export_edf_btn = QtWidgets.QPushButton("→ EDF (clínico)")
        self.export_edf_btn.setToolTip("European Data Format (clínico, lido por Persyst/Polysmith/EEGLAB)")
        self.export_edf_btn.clicked.connect(self._export_to_edf)
        if not HAS_EDF:
            self.export_edf_btn.setEnabled(False)
            self.export_edf_btn.setText("EDF indisponível (pip install pyedflib)")
        eg.addWidget(self.export_edf_btn, 0, 1)
        self.export_fif_btn = QtWidgets.QPushButton("→ FIF (MNE-Python)")
        self.export_fif_btn.setToolTip("FIF (MNE/Brainstorm/FieldTrip — análise científica)")
        self.export_fif_btn.clicked.connect(self._export_to_fif)
        if not HAS_MNE:
            self.export_fif_btn.setEnabled(False)
            self.export_fif_btn.setText("FIF indisponível (pip install mne)")
        eg.addWidget(self.export_fif_btn, 0, 2)
        self.export_pdf_btn = QtWidgets.QPushButton("→ Relatório PDF")
        self.export_pdf_btn.setToolTip("PDF com resumo da sessão + figuras")
        self.export_pdf_btn.clicked.connect(self._export_pdf_report)
        if not (HAS_REPORTLAB and HAS_MPL):
            self.export_pdf_btn.setEnabled(False)
            self.export_pdf_btn.setText("PDF indisponível (matplotlib/reportlab)")
        eg.addWidget(self.export_pdf_btn, 0, 3)
        # Linha 2: BIDS export (padrão científico)
        self.export_bids_btn = QtWidgets.QPushButton("→ BIDS-EEG (padrão científico)")
        self.export_bids_btn.setToolTip(
            "Brain Imaging Data Structure — padrão de organização "
            "para dados de neuroimagem reprodutíveis."
        )
        self.export_bids_btn.clicked.connect(self._export_to_bids)
        eg.addWidget(self.export_bids_btn, 1, 0, 1, 2)

        eg_info = QtWidgets.QLabel(
            "Cada exportação abre um diálogo: escolha o data.csv da sessão a exportar. "
            "Os arquivos são gravados na mesma pasta da sessão. "
            "<b>BIDS</b> cria estrutura <code>bids/sub-NN/ses-MM/eeg/</code> "
            "interoperável com EEGLAB, MNE, FieldTrip e Brainstorm."
        )
        eg_info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        eg_info.setStyleSheet(f"color: {COLORS['text_dim']};"); eg_info.setWordWrap(True)
        eg.addWidget(eg_info, 2, 0, 1, 4)
        layout.addWidget(export_group)

        layout.addStretch()
        sub_tabs.addTab(sess_page, "Sessão e Arquivos")

        # ============== Sub-aba: CAMINHOS ==============
        path_page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(path_page)
        layout.setContentsMargins(10, 8, 10, 8); layout.setSpacing(8)

        path_group = QtWidgets.QGroupBox("Caminhos do Sistema (editáveis)")
        pgl = QtWidgets.QVBoxLayout(path_group)
        info_lbl = QtWidgets.QLabel(
            "Você pode editar diretamente ou usar 'Procurar...'. "
            "O botão 'Padrão' restaura o valor original. "
            "Mudanças em caminhos do sistema só passam a valer após reiniciar."
        )
        info_lbl.setStyleSheet(f"color: {COLORS['text_dim']}; font-style: italic;")
        info_lbl.setWordWrap(True)
        pgl.addWidget(info_lbl)

        # Cada linha: rótulo + QLineEdit + Procurar + Padrão + Abrir
        self._path_editors = {}
        # Tuplas: (chave, rótulo, valor_atual, valor_padrao, editável_em_runtime)
        # CONFIG_PATH e SCRIPT_DIR são informativos (não podem ser alterados em runtime).
        path_rows = [
            ("config_path",   "Config (JSON):",         CONFIG_PATH, CONFIG_PATH, False),
            ("script_dir",    "Diretório do script:",   SCRIPT_DIR,  SCRIPT_DIR,  False),
            ("doc_dir",       "Documents/EEG_Coletor:", DOC_DIR,     DOC_DIR,     False),
            ("save_dir",      "Pasta de salvamento:",   self.config.save_directory,
                                                       DEFAULT_SAVE_DIRECTORY, True),
        ]
        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(6); grid.setVerticalSpacing(4)
        for row, (key, label, cur_val, default_val, editable) in enumerate(path_rows):
            lbl = QtWidgets.QLabel(label)
            grid.addWidget(lbl, row, 0)
            edit = QtWidgets.QLineEdit(cur_val)
            edit.setReadOnly(not editable)
            edit.setProperty("default_value", default_val)
            edit.setProperty("path_key", key)
            if not editable:
                edit.setToolTip("Caminho do sistema — somente leitura (use 'Abrir' ou 'Copiar')")
            else:
                edit.setToolTip("Edite diretamente ou use 'Procurar...'. 'Padrão' restaura o valor original.")
                edit.editingFinished.connect(
                    lambda k=key: self._on_system_path_edited(k))
            grid.addWidget(edit, row, 1)
            self._path_editors[key] = edit

            # Botão Procurar (só para editáveis)
            if editable:
                btn_browse = QtWidgets.QPushButton("Procurar...")
                btn_browse.clicked.connect(lambda _c, k=key: self._browse_system_path(k))
                grid.addWidget(btn_browse, row, 2)
            else:
                grid.addWidget(QtWidgets.QLabel(""), row, 2)

            # Botão Padrão
            btn_default = QtWidgets.QPushButton("Padrão")
            btn_default.setToolTip(f"Restaurar para: {default_val}")
            btn_default.setEnabled(editable)
            btn_default.clicked.connect(lambda _c, k=key: self._reset_system_path(k))
            grid.addWidget(btn_default, row, 3)

            # Botão Abrir pasta
            btn_open = QtWidgets.QPushButton("Abrir")
            btn_open.setToolTip("Abre o caminho no explorador de arquivos")
            btn_open.clicked.connect(lambda _c, k=key: self._open_system_path(k))
            grid.addWidget(btn_open, row, 4)

            # Botão Copiar
            btn_copy = QtWidgets.QPushButton("Copiar")
            btn_copy.setToolTip("Copia o caminho para a área de transferência")
            btn_copy.clicked.connect(lambda _c, k=key: self._copy_system_path(k))
            grid.addWidget(btn_copy, row, 5)

        # Colunas que esticam (QLineEdit) vs fixas (botões)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        for c in (2, 3, 4, 5):
            grid.setColumnStretch(c, 0)
        pgl.addLayout(grid)
        layout.addWidget(path_group)

        # Log de auditoria
        audit_group = QtWidgets.QGroupBox("Log de Auditoria (events.jsonl)")
        agl = QtWidgets.QVBoxLayout(audit_group)
        agl.addWidget(QtWidgets.QLabel(
            "Cada acao do operador é registrada em <pasta_sessao>/events.jsonl "
            "com timestamp + ação + valor antes/depois. Útil para validação "
            "regulatoria (FDA 21 CFR Part 11 / LGPD)."
        ))
        layout.addWidget(audit_group)
        layout.addStretch()
        sub_tabs.addTab(path_page, "Caminhos e Auditoria")

        return outer

    # ---- Caminhos do sistema (editores) ----
    def _on_system_path_edited(self, key):
        """Chamado quando o usuário termina de editar um caminho."""
        if not hasattr(self, "_path_editors"): return
        edit = self._path_editors.get(key)
        if edit is None: return
        new_val = edit.text().strip()
        if not new_val: return
        if key == "save_dir":
            # Aplica como save_directory (com criação de pasta + validação)
            try:
                os.makedirs(new_val, exist_ok=True)
                self.config.save_directory = new_val
                self.config.save()
                # Reflete no campo de salvamento da outra sub-aba, se existir
                if hasattr(self, "savedir_label"):
                    self.savedir_label.setText(new_val)
                self._log(f"Pasta de salvamento atualizada: {new_val}")
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self, "Caminho inválido",
                    f"Não foi possível usar '{new_val}':\n{exc}"
                )
                edit.setText(self.config.save_directory)

    def _browse_system_path(self, key):
        """Abre diálogo de pasta e atualiza o campo correspondente."""
        edit = self._path_editors.get(key) if hasattr(self, "_path_editors") else None
        if edit is None: return
        cur = edit.text().strip() or os.path.expanduser("~")
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Escolher pasta", cur)
        if d:
            edit.setText(d)
            self._on_system_path_edited(key)

    def _reset_system_path(self, key):
        """Restaura um caminho ao seu valor padrão."""
        edit = self._path_editors.get(key) if hasattr(self, "_path_editors") else None
        if edit is None: return
        default = edit.property("default_value")
        if default:
            edit.setText(default)
            self._on_system_path_edited(key)

    def _open_system_path(self, key):
        """Abre o caminho no explorador de arquivos do sistema."""
        edit = self._path_editors.get(key) if hasattr(self, "_path_editors") else None
        if edit is None: return
        path = edit.text().strip()
        if not path: return
        # Se for arquivo, abre a pasta que o contém
        target = path if os.path.isdir(path) else os.path.dirname(path)
        try:
            if sys.platform == "win32":
                os.startfile(target)  # type: ignore
            elif sys.platform == "darwin":
                import subprocess
                subprocess.Popen(["open", target])
            else:
                import subprocess
                subprocess.Popen(["xdg-open", target])
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Não foi possível abrir",
                f"Erro ao abrir '{target}':\n{exc}"
            )

    def _copy_system_path(self, key):
        """Copia o caminho para a área de transferência."""
        edit = self._path_editors.get(key) if hasattr(self, "_path_editors") else None
        if edit is None: return
        path = edit.text().strip()
        if not path: return
        cb = QtWidgets.QApplication.clipboard()
        cb.setText(path)
        if hasattr(self, "status_state_lbl"):
            self.status_state_lbl.setText("Caminho copiado")
        self._log(f"Caminho copiado: {path}")

    # ---- Idioma (i18n) ----
    def _on_language_changed(self, *_args):
        """Aplica idioma escolhido (re-titula abas + salva preferencia)."""
        if not hasattr(self, "lang_combo"): return
        code = self.lang_combo.currentData()
        if not code or code == getattr(self.config, "language", "pt"):
            return
        I18N.set_language(code)
        self.config.language = code
        self.config.save()
        self._retranslate_visible_ui()
        QtWidgets.QMessageBox.information(
            self, "Idioma alterado",
            "O idioma foi alterado. Alguns textos hardcoded só serão "
            "totalmente atualizados após reiniciar o aplicativo."
        )

    def _retranslate_visible_ui(self):
        """Re-aplica tr() em widgets cujo texto está no dicionário I18N.

        Estratégia:
            1. Constrói um mapa REVERSO de cada idioma de volta para pt-BR.
            2. Para cada widget visível com texto, tenta:
               - Se o texto já está em pt (chave conhecida), aplica tr().
               - Se está em en/es (valor conhecido), reverte para pt e aplica tr().
            3. Percorre QTabWidget (tabs), QGroupBox, QPushButton, QLabel,
               QCheckBox, QRadioButton, QAction recursivamente.
        """
        # ===== 1) Mapa reverso (valor traduzido -> chave em pt) =====
        rev_to_pt = {}
        for lang_code in ("en", "es"):
            lang_map = I18N._maps.get(lang_code, {})
            for pt_key, translated in lang_map.items():
                # Não sobrescreve se idêntico ao pt (evita loop)
                if translated and translated != pt_key:
                    rev_to_pt[translated] = pt_key
        # Chaves em pt (todos os textos que sabemos traduzir)
        known_pt_keys = set()
        for lm in (I18N._en, I18N._es):
            known_pt_keys.update(lm.keys())

        def _try_translate(text):
            """Retorna o texto traduzido para o idioma atual, ou None se não souber."""
            if not isinstance(text, str) or not text:
                return None
            if text in known_pt_keys:
                return tr(text)            # já em pt → aplica
            if text in rev_to_pt:
                return tr(rev_to_pt[text]) # em en/es → reverte e aplica
            return None

        # ===== 2) Abas top-level + sub-abas =====
        # Mantemos a abordagem por índice para tabs (mais robusta)
        if hasattr(self, "tabs"):
            top = ["Configurar", "Visualizar", "Analisar", "Sistema"]
            for i, key in enumerate(top):
                if i < self.tabs.count():
                    self.tabs.setTabText(i, tr(key))
        try:
            setup_titles = ["Voluntários", "Conexão", "Filtros e Canais",
                            "Hardware", "Calibração"]
            view_titles  = ["Tempo Real", "Topografia", "Espectrograma",
                            "Bio (EMG/ECG/EoG)", "Histórico", "Layout Custom"]
            anal_titles  = ["Análises", "Offline", "ERP", "Conectividade",
                            "ERS/ERD", "Focus / SSVEP", "EMG Joystick",
                            "BCI Trainer (MI)"]
            sys_titles   = ["Rede e Eventos", "Configurações"]
            for key, titles in (
                ("setup", setup_titles), ("view", view_titles),
                ("analyse", anal_titles), ("system", sys_titles)
            ):
                sub = self._sub_tabs.get(key) if hasattr(self, "_sub_tabs") else None
                if sub is None: continue
                for i, t in enumerate(titles):
                    if i < sub.count():
                        sub.setTabText(i, tr(t))
            # Sub-sub-abas Bio
            bio_titles = ["EMG · Músculos", "ECG · Coração", "EoG · Olhos"]
            if hasattr(self, "bio_tabs"):
                for i, t in enumerate(bio_titles):
                    if i < self.bio_tabs.count():
                        # Esses títulos não estão no dicionário — usa tr() puro
                        self.bio_tabs.setTabText(i, tr(t))
        except Exception:
            pass

        # ===== 3) Walker recursivo: QGroupBox, QPushButton, QLabel,
        #         QCheckBox, QRadioButton, QAction =====
        try:
            for gb in self.findChildren(QtWidgets.QGroupBox):
                new = _try_translate(gb.title())
                if new is not None and new != gb.title():
                    gb.setTitle(new)
        except Exception: pass

        try:
            for btn in self.findChildren(QtWidgets.QAbstractButton):
                # Pula botões muito pequenos / sem texto / ícones puros
                if not btn.text(): continue
                new = _try_translate(btn.text())
                if new is not None and new != btn.text():
                    btn.setText(new)
        except Exception: pass

        try:
            for lbl in self.findChildren(QtWidgets.QLabel):
                if not lbl.text(): continue
                # Skip labels com HTML complexo
                if "<" in lbl.text() and ">" in lbl.text():
                    continue
                new = _try_translate(lbl.text())
                if new is not None and new != lbl.text():
                    lbl.setText(new)
        except Exception: pass

        try:
            menu_bar = self.menuBar() if hasattr(self, "menuBar") else None
            if menu_bar:
                for menu in menu_bar.findChildren(QtWidgets.QMenu):
                    new_title = _try_translate(menu.title())
                    if new_title is not None and new_title != menu.title():
                        menu.setTitle(new_title)
                for act in menu_bar.findChildren(QtGui.QAction):
                    new_text = _try_translate(act.text())
                    if new_text is not None and new_text != act.text():
                        act.setText(new_text)
        except Exception: pass

    # ---- Logo UFES adaptado ao tema ----
    def _refresh_ufes_logo_pixmap(self):
        """Exibe o logo UFES de modo que ACOMPANHE o fundo do header.

        O brasão é circular e o PNG vem com os CANTOS pretos. Aqui o fundo
        preto conectado às bordas vira transparente (uma única vez, em cache),
        preservando o miolo (textos/contornos pretos cercados pelo anel
        branco). Só redimensiona — sem inversão/recoloração por tema."""
        if not getattr(self, "ufes_logo_lbl", None):
            return
        if not os.path.exists(LOGO_UFES_PATH):
            return
        base = getattr(self, "_ufes_base_pm", None)
        if base is None or base.isNull():
            pm = QtGui.QPixmap(LOGO_UFES_PATH)
            if pm.isNull():
                return
            base = self._logo_clear_bg_corners(pm)
            self._ufes_base_pm = base
        pm = base.scaledToHeight(
            40, QtCore.Qt.TransformationMode.SmoothTransformation)
        self.ufes_logo_lbl.setPixmap(pm)

    @staticmethod
    def _logo_clear_bg_corners(pm):
        """Torna transparente o fundo conectado às bordas (cantos pretos do
        brasão). Idempotente e seguro: se já houver alfa transparente, se não
        houver fundo escuro, ou se algo falhar, devolve o pixmap original."""
        try:
            img = pm.toImage().convertToFormat(
                QtGui.QImage.Format.Format_ARGB32)
            w, h = img.width(), img.height()
            if w < 2 or h < 2:
                return pm
            if img.pixelColor(0, 0).alpha() == 0:
                return pm  # já tem fundo transparente
            import numpy as _np
            from scipy import ndimage as _ndi
            arr = _np.frombuffer(
                img.constBits(), _np.uint8).reshape(h, w, 4).copy()  # BGRA
            soma = (arr[..., 0].astype(_np.int16)
                    + arr[..., 1] + arr[..., 2])
            nearblack = (soma < 120) & (arr[..., 3] > 0)
            if not nearblack.any():
                return pm
            lbl, n = _ndi.label(nearblack)
            if n == 0:
                return pm
            border = _np.unique(_np.concatenate(
                [lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]]))
            border = border[border != 0]
            if border.size == 0:
                return pm
            arr[_np.isin(lbl, border), 3] = 0
            out = QtGui.QImage(arr.tobytes(), w, h, w * 4,
                               QtGui.QImage.Format.Format_ARGB32).copy()
            return QtGui.QPixmap.fromImage(out)
        except Exception:
            return pm

    # ---- Handlers do tema ----
    def _on_theme_changed(self, theme_name):
        if theme_name not in THEMES:
            return
        self.config.theme = theme_name
        _apply_theme_colors(theme_name)
        # Atualiza stylesheet globalmente
        app = QtWidgets.QApplication.instance()
        if app:
            app.setStyleSheet(build_stylesheet(COLORS))
        # pyqtgraph background — setConfigOption só afeta gráficos NOVOS, então
        # atualizamos explicitamente TODOS os plots já criados (senão o "fundo
        # da tela" dos gráficos não acompanha a troca de tema).
        pg.setConfigOption("background", COLORS["background"])
        pg.setConfigOption("foreground", COLORS["text"])
        try:
            for cls in (pg.PlotWidget, pg.GraphicsLayoutWidget, pg.ImageView):
                for w in self.findChildren(cls):
                    try:
                        w.setBackground(COLORS["background"])
                    except Exception:
                        pass
        except Exception:
            pass
        # Reaplica estilos inline em widgets-chave
        self._reapply_themed_inline_styles()
        # Logo UFES (re-processa cor de fundo)
        self._refresh_ufes_logo_pixmap()
        # Repinta widgets customizados
        if hasattr(self, "head_plot"):
            self.head_plot.setStyleSheet(f"background-color: {COLORS['surface']};")
            self.head_plot.update()
        # Repopula os color pickers do editor com as cores do tema escolhido
        self._reload_color_pickers_from_active()
        # Atualiza visibilidade do botão Deletar (so habilitado em tema custom)
        self._refresh_delete_theme_button()
        self.config.save()
        self._log(f"Tema aplicado: {theme_name}")
        self._audit_event("theme_change", theme=theme_name)

    def _refresh_delete_theme_button(self):
        if not hasattr(self, "delete_theme_btn"): return
        cur = self.theme_combo.currentText() if hasattr(self, "theme_combo") else ""
        is_custom = cur in self.config.custom_themes
        self.delete_theme_btn.setEnabled(is_custom)
        self.delete_theme_btn.setToolTip(
            "Apenas temas personalizados podem ser deletados"
            if not is_custom else f"Remove permanentemente '{cur}'"
        )

    def _delete_current_custom_theme(self):
        """Remove o tema personalizado atualmente selecionado."""
        if not hasattr(self, "theme_combo"): return
        cur = self.theme_combo.currentText()
        if cur not in self.config.custom_themes:
            self._log("Apenas temas personalizados podem ser deletados", error=True)
            return
        confirm = QtWidgets.QMessageBox.question(
            self, "Deletar tema personalizado",
            f"Confirmar exclusao do tema personalizado '{cur}'?",
            QtWidgets.QMessageBox.StandardButton.Yes |
            QtWidgets.QMessageBox.StandardButton.No
        )
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        # Remove de THEMES (em memoria) e do config (persistido)
        THEMES.pop(cur, None)
        self.config.custom_themes.pop(cur, None)
        # Remove do combo e seleciona um padrão
        idx = self.theme_combo.findText(cur)
        if idx >= 0:
            self.theme_combo.blockSignals(True)
            self.theme_combo.removeItem(idx)
            self.theme_combo.blockSignals(False)
        # Pega o primeiro tema disponivel (preferindo Lime)
        fallback = "Lime (verde-limao)" if "Lime (verde-limao)" in THEMES else next(iter(THEMES))
        self.theme_combo.setCurrentText(fallback)
        self.config.theme = fallback
        self.config.save()
        self._log(f"Tema '{cur}' deletado. Aplicando '{fallback}'.")

    def _reapply_themed_inline_styles(self):
        """Reestiliza TODOS os widgets cuja cor foi setada inline.
        Inclui header, titulo, indicadores, tabelas, head plot, etc."""
        # ----- Header e titulo -----
        if hasattr(self, "header_widget"):
            self.header_widget.setStyleSheet(
                f"#header {{ background-color: {COLORS['surface']}; "
                f"border: 1px solid {COLORS['border']}; border-radius: 4px; }}")
        if hasattr(self, "title_label"):
            self.title_label.setStyleSheet(
                f"color: {COLORS['accent']}; background: transparent; font-size: 18pt; "
                f"font-weight: bold; letter-spacing: 2px; font-family: {FONT_UI_STACK};")
        if hasattr(self, "expansion_label"):
            n = getattr(self, "num_channels", BASE_CHANNELS)
            if n > BASE_CHANNELS:
                self.expansion_label.setStyleSheet(
                    f"color: {COLORS['expansion']}; font-weight: bold; padding: 0 6px;"
                    f"border: 1px solid {COLORS['expansion']}; border-radius: 3px;")
            else:
                self.expansion_label.setStyleSheet(
                    f"color: {COLORS['text_dim']}; font-weight: bold; padding: 0 6px;"
                    f"border: 1px solid {COLORS['border']}; border-radius: 3px;")
        # ----- Indicadores de status -----
        connected = (self.serial_thread is not None
                     and self.serial_thread.isRunning())
        col = COLORS["accent"] if connected else COLORS["error"]
        if hasattr(self, "status_dot"):
            self.status_dot.setStyleSheet(f"color: {col}; font-size: 22pt;")
        if hasattr(self, "status_label"):
            self.status_label.setStyleSheet(
                f"color: {col}; font-weight: bold; padding: 0 14px; font-size: 11pt;")
        if hasattr(self, "samples_label"):
            self.samples_label.setStyleSheet(
                f"color: {COLORS['text_dim']}; padding: 0 10px; "
                f"font-family: {FONT_DATA_STACK};")
        if hasattr(self, "accel_label"):
            self.accel_label.setStyleSheet(
                f"color: {COLORS['text_dim']}; padding: 0 10px; "
                f"font-family: {FONT_DATA_STACK};")
        if hasattr(self, "rec_indicator"):
            txt = "● REC" if self.is_recording else ""
            self.rec_indicator.setText(txt)
            self.rec_indicator.setStyleSheet(
                f"color: {COLORS['error']}; font-weight: bold;")
        # ----- Nome dos canais nas tabelas (cor por canal) -----
        if hasattr(self, "stats_table"):
            for ch in range(MAX_CHANNELS):
                item = self.stats_table.item(ch, 0)
                if item is not None:
                    item.setForeground(QtGui.QColor(CHANNEL_COLORS[ch]))
        if hasattr(self, "imp_table"):
            for ch in range(MAX_CHANNELS):
                item = self.imp_table.item(ch, 0)
                if item is not None:
                    item.setForeground(QtGui.QColor(CHANNEL_COLORS[ch]))
        # ----- Head plot e widgets customizados -----
        if hasattr(self, "head_plot"):
            self.head_plot.setStyleSheet(f"background-color: {COLORS['surface']};")
            self.head_plot.update()
        # ----- Checkboxes coloridos dos canais -----
        if hasattr(self, "channel_checks"):
            for ch, cb in enumerate(self.channel_checks):
                cb.setStyleSheet(
                    f"QCheckBox {{ color: {CHANNEL_COLORS[ch]}; font-weight: bold; }}")
        if hasattr(self, "map_combos"):
            # Labels dos CHs na aba Configurações (se existirem como widgets siblings)
            pass
        # ----- Atualiza palette do QApplication para Qt nativo (Base, AlternateBase) -----
        app = QtWidgets.QApplication.instance()
        if app:
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.ColorRole.Window,          QtGui.QColor(COLORS["background"]))
            palette.setColor(QtGui.QPalette.ColorRole.WindowText,      QtGui.QColor(COLORS["text"]))
            palette.setColor(QtGui.QPalette.ColorRole.Base,            QtGui.QColor(COLORS["surface_alt"]))
            palette.setColor(QtGui.QPalette.ColorRole.AlternateBase,   QtGui.QColor(COLORS["surface"]))
            palette.setColor(QtGui.QPalette.ColorRole.Text,            QtGui.QColor(COLORS["text"]))
            palette.setColor(QtGui.QPalette.ColorRole.Button,          QtGui.QColor(COLORS["surface_alt"]))
            palette.setColor(QtGui.QPalette.ColorRole.ButtonText,      QtGui.QColor(COLORS["accent"]))
            palette.setColor(QtGui.QPalette.ColorRole.Highlight,       QtGui.QColor(COLORS["accent_dim"]))
            palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(COLORS["background"]))
            palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase,     QtGui.QColor(COLORS["surface"]))
            palette.setColor(QtGui.QPalette.ColorRole.ToolTipText,     QtGui.QColor(COLORS["text"]))
            app.setPalette(palette)
        # forca repaint
        self.update()
        self.repaint()

    # ---- Mapeamento ----
    def _on_mapping_changed(self, _text=None):
        # Atualiza config em memoria e Head Plot ao vivo
        for ch in range(MAX_CHANNELS):
            self.config.channel_mapping[ch] = self.map_combos[ch].currentText()
        if hasattr(self, "head_plot"):
            self.head_plot.set_mapping(self.config.channel_mapping)

    def _save_mapping_and_apply(self):
        self._on_mapping_changed()
        self.config.save()
        self._log(f"Mapeamento salvo: {self.config.channel_mapping[:self.num_channels]}")

    def _reset_mapping_to_default(self):
        self.config.channel_mapping = list(DEFAULT_MAPPING)
        for ch in range(MAX_CHANNELS):
            idx = self.map_combos[ch].findText(DEFAULT_MAPPING[ch])
            if idx >= 0:
                self.map_combos[ch].blockSignals(True)
                self.map_combos[ch].setCurrentIndex(idx)
                self.map_combos[ch].blockSignals(False)
        if hasattr(self, "head_plot"):
            self.head_plot.set_mapping(self.config.channel_mapping)
        self.config.save()
        self._log("Mapeamento restaurado para o padrão Fp1/Fp2/C3/C4/...")

    # ---- Sessão / snapshots ----
    def _on_subject_changed(self, text):
        self.config.subject = text.strip()
        self._refresh_session_preview()

    def _on_template_changed(self, text):
        self.config.session_template = text
        self._refresh_session_preview()

    def _on_snapshot_changed(self, val):
        self.config.snapshot_interval = int(val)

    def _save_all_settings(self):
        self.config.save()
        self._log("Configurações salvas em config.json")

    def _build_session_name(self):
        now = datetime.now()
        try:
            return self.config.session_template.format(
                subject=self.config.subject or "sujeito",
                date=now.strftime("%Y-%m-%d"),
                time=now.strftime("%H-%M-%S"),
                datetime=now.strftime("%Y-%m-%d_%H-%M-%S"),
                channels=self.num_channels,
            )
        except Exception:
            return now.strftime("EEG_%Y-%m-%d_%H-%M-%S")

    def _refresh_session_preview(self):
        if hasattr(self, "session_preview"):
            self.session_preview.setText(self._build_session_name() + "/")

    # ==================================================================
    # Pasta de salvamento (configuravel)
    # ==================================================================
    def _on_savedir_text_changed(self):
        txt = self.savedir_label.text().strip()
        if not txt:
            return
        self._set_save_directory(txt)

    def _browse_save_directory(self):
        start = self.config.save_directory if os.path.isdir(self.config.save_directory) else SCRIPT_DIR
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Escolher pasta de salvamento das sessões", start)
        if d:
            self._set_save_directory(d)

    def _reset_save_directory_to_default(self):
        self._set_save_directory(DEFAULT_SAVE_DIRECTORY)

    def _set_save_directory(self, path):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as exc:
            self._log(f"Não foi possivel criar/usar a pasta: {path} ({exc})", error=True)
            return
        self.config.save_directory = path
        self.config.save()
        # Atualiza widgets que mostram o caminho
        if hasattr(self, "savedir_label"):
            self.savedir_label.blockSignals(True)
            self.savedir_label.setText(path)
            self.savedir_label.blockSignals(False)
        if hasattr(self, "conn_save_dir_label"):
            self.conn_save_dir_label.setText(path)
        self._log(f"Pasta de salvamento: {path}")

    # ==================================================================
    # Editor de cores personalizado
    # ==================================================================
    def _pick_color(self, key):
        """Abre QColorDialog para a chave indicada."""
        current = QtGui.QColor(COLORS[key])
        col = QtWidgets.QColorDialog.getColor(
            current, self, f"Cor: {key}",
            QtWidgets.QColorDialog.ColorDialogOption.DontUseNativeDialog,
        )
        if col.isValid():
            hexc = col.name()  # "#rrggbb"
            self.color_edits[key].setText(hexc)
            self._set_color_button(key, hexc)

    def _hex_color_changed(self, key):
        """Quando o usuário edita o campo de texto hex."""
        txt = self.color_edits[key].text().strip()
        if not txt.startswith("#"): txt = "#" + txt
        col = QtGui.QColor(txt)
        if col.isValid():
            self._set_color_button(key, col.name())
        else:
            # restaura
            self.color_edits[key].setText(COLORS[key])

    def _set_color_button(self, key, hex_color):
        if key in self.color_buttons:
            self.color_buttons[key].setStyleSheet(
                f"background-color: {hex_color}; border: 1px solid {COLORS['border']};")

    def _apply_custom_colors(self):
        """Aplica as cores atualmente nos campos como tema 'ao vivo'."""
        new_colors = {}
        for key, edit in self.color_edits.items():
            txt = edit.text().strip()
            if not txt.startswith("#"): txt = "#" + txt
            col = QtGui.QColor(txt)
            new_colors[key] = col.name() if col.isValid() else COLORS[key]
        # mutacao in-place de COLORS
        for k, v in new_colors.items():
            COLORS[k] = v
        # Re-aplica stylesheet
        app = QtWidgets.QApplication.instance()
        if app:
            app.setStyleSheet(build_stylesheet(COLORS))
        pg.setConfigOption("background", COLORS["background"])
        pg.setConfigOption("foreground", COLORS["text"])
        self._reapply_themed_inline_styles()
        self._log("Cores personalizadas aplicadas (ao vivo)")

    def _save_custom_theme(self):
        """Salva as cores atuais como um tema nomeado em THEMES e em config.json."""
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Salvar tema", "Nome do tema personalizado:",
            QtWidgets.QLineEdit.EchoMode.Normal, "Meu Tema"
        )
        if not ok or not name.strip(): return
        name = name.strip()
        # garante que aplicou primeiro
        self._apply_custom_colors()
        # registra em THEMES
        THEMES[name] = dict(COLORS)
        # adiciona ao combo da aba se ainda não tiver
        if self.theme_combo.findText(name) < 0:
            self.theme_combo.addItem(name)
        # seleciona
        self.theme_combo.setCurrentText(name)
        self.config.theme = name
        # Persistencia: AppConfig agora salva também temas customizados
        self.config.custom_themes[name] = dict(COLORS)
        self.config.save()
        self._log(f"Tema personalizado '{name}' salvo. Use o combo para alternar.")

    def _reload_color_pickers_from_active(self):
        """Volta os campos para o tema atualmente ativo."""
        for key, edit in self.color_edits.items():
            edit.blockSignals(True)
            edit.setText(COLORS[key])
            edit.blockSignals(False)
            self._set_color_button(key, COLORS[key])


# ============================================================
# Termo de Uso + Assistente de primeiro uso (setup)
# ============================================================
# Termo embutido (fallback) — usado se TERMO_DE_USO.md/.txt nao for encontrado
# em disco. O texto COMPLETO fica em TERMO_DE_USO.md (ao lado do app), carregado
# por _load_terms_text(); este resumo cobre as clausulas essenciais.
TERMS_TEXT_FALLBACK = """\
# Termo de Consentimento e Uso do Software — OpenBiônica

Leia antes de usar. Ao prosseguir, você declara que leu, entendeu e concorda.
Se não concordar, não utilize o software.

## 1. O que é
Software **open source** e **sem fins lucrativos** (projeto OpenBionica) para
coleta, visualização e análise de biossinais (EEG, EMG, ECG, EoG, acelerômetro),
destinado a **pesquisa e educação**.

## 2. NÃO é dispositivo médico
**NÃO é dispositivo médico** e **não foi certificado** por qualquer autoridade.
**Não use para diagnóstico, tratamento ou decisão clínica.** Sinais e análises
têm caráter exploratório/educativo e podem conter ruído e erros.

## 3. Privacidade — funciona offline
O software processa os dados **exclusivamente de forma LOCAL**, na sua máquina
(pastas `sessions/` e `Documentos/EEG_Coletor`). **NÃO coleta, NÃO transmite,
NÃO compartilha e NÃO "rouba" dados.** O autor **não tem acesso** a nenhum dado.
A única função de rede é a verificação **manual e opcional** de atualização, que
apenas **baixa código** (nunca envia dados) e pode ficar desligada — por padrão,
o software funciona **offline**.

## 4. Responsabilidade de quem usa (LGPD)
Se você usar o software para coletar dados de terceiros (participantes), **VOCÊ é
o controlador** desses dados (LGPD, Lei 13.709/2018) e o único responsável por
obter o consentimento dos participantes, pela guarda/segurança/anonimização e
pelo cumprimento legal/ético. Biossinais de pessoas identificáveis são, em regra,
**dados sensíveis**. O autor **não é controlador nem operador** desses dados.

## 5. Isenção de garantia e limitação de responsabilidade
Software fornecido **"COMO ESTÁ"**, sem garantias de qualquer espécie. Na máxima
extensão permitida por lei, o autor **não se responsabiliza** por danos diretos
ou indiretos (incluindo perda de dados e decisões tomadas com base nos
resultados). Você usa por sua conta e risco e deve manter backups.

## 6. Licença
Distribuído sob a licença **MIT** (ver arquivo LICENSE). Componentes de terceiros
têm suas próprias licenças.

## 7. Aceite
Ao marcar "Li e concordo" e concluir, você aceita este termo. A versão do termo e
a data do aceite ficam registradas **apenas no seu computador**.

> Modelo de termo — não constitui aconselhamento jurídico. O texto completo está
> em TERMO_DE_USO.md.
"""


def _load_terms_text():
    """Carrega o termo COMPLETO de disco (ao lado do app ou em DOC_DIR);
    cai no resumo embutido se nao encontrar."""
    cands = [os.path.join(SCRIPT_DIR, "TERMO_DE_USO.md"),
             os.path.join(SCRIPT_DIR, "TERMO_DE_USO.txt"),
             os.path.join(DOC_DIR,    "TERMO_DE_USO.md")]
    if _MEIPASS_DIR:                       # arquivos embutidos no .exe (_internal)
        cands += [os.path.join(_MEIPASS_DIR, "TERMO_DE_USO.md"),
                  os.path.join(_MEIPASS_DIR, "TERMO_DE_USO.txt")]
    for p in cands:
        try:
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    return f.read()
        except Exception:
            pass
    return TERMS_TEXT_FALLBACK


# Presets de layout do assistente (codigos validos de PANEL_KINDS:
# empty, ts1, fft, bands, head, spec, accel, focus)
WIZARD_LAYOUT_PRESETS = {
    "Padrão (recomendado)": ["ts1", "fft", "head", "bands"],
    "Foco EEG":             ["ts1", "fft", "spec", "bands"],
    "EMG / Biossinais":     ["ts1", "ts1", "accel", "focus"],
    "Minimalista":          ["ts1", "fft", "empty", "empty"],
}


class FirstRunWizard(QtWidgets.QDialog):
    """Assistente de primeiro uso: idioma -> layout -> aceite do Termo de Uso.
    Grava no AppConfig recebido e chama config.save() ao concluir. Recusar o
    termo => reject() => o app sai no gate do main()."""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self._chosen_layout = "Padrão (recomendado)"
        self.setWindowTitle(f"Bem-vindo ao {APP_NAME}")
        self.setModal(True)
        self.setMinimumSize(680, 600)
        root = QtWidgets.QVBoxLayout(self)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._build_page_language())  # 0
        self.stack.addWidget(self._build_page_layout())    # 1
        self.stack.addWidget(self._build_page_terms())     # 2
        root.addWidget(self.stack, 1)
        nav = QtWidgets.QHBoxLayout()
        self.btn_decline = QtWidgets.QPushButton("Recusar e sair")
        self.btn_back    = QtWidgets.QPushButton("Voltar")
        self.btn_next    = QtWidgets.QPushButton("Avançar")
        self.btn_decline.clicked.connect(self.reject)
        self.btn_back.clicked.connect(self._go_back)
        self.btn_next.clicked.connect(self._go_next)
        nav.addWidget(self.btn_decline); nav.addStretch(1)
        nav.addWidget(self.btn_back); nav.addWidget(self.btn_next)
        root.addLayout(nav)
        try:
            self.setStyleSheet(build_stylesheet(COLORS))
        except Exception:
            pass
        self._refresh_nav()

    def _build_page_language(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        v.addWidget(QtWidgets.QLabel("<h2>1. Idioma</h2>"))
        v.addWidget(QtWidgets.QLabel("Escolha o idioma da interface:"))
        self.lang_combo = QtWidgets.QComboBox()
        for code, label in I18N.LANGUAGES.items():
            self.lang_combo.addItem(label, code)
        idx = self.lang_combo.findData(getattr(self.config, "language", "pt"))
        self.lang_combo.setCurrentIndex(max(0, idx))
        self.lang_combo.currentIndexChanged.connect(self._on_lang)
        v.addWidget(self.lang_combo)
        v.addWidget(QtWidgets.QLabel(
            "<i>Alguns textos só mudam de idioma após reiniciar o programa.</i>"))
        v.addStretch(1)
        return w

    def _on_lang(self):
        code = self.lang_combo.currentData()
        if code:
            try: I18N.set_language(code)
            except Exception: pass
            self.config.language = code

    def _build_page_layout(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        v.addWidget(QtWidgets.QLabel("<h2>2. Layout dos painéis</h2>"))
        v.addWidget(QtWidgets.QLabel(
            "Organização inicial dos 4 painéis (mude depois em "
            "Visualizar → Layout Custom):"))
        self._layout_group = QtWidgets.QButtonGroup(self)
        for name, kinds in WIZARD_LAYOUT_PRESETS.items():
            rb = QtWidgets.QRadioButton(f"{name}   ({', '.join(kinds)})")
            rb.setProperty("preset", name)
            if name == self._chosen_layout: rb.setChecked(True)
            rb.toggled.connect(self._on_layout)
            self._layout_group.addButton(rb)
            v.addWidget(rb)
        v.addStretch(1)
        return w

    def _on_layout(self):
        for b in self._layout_group.buttons():
            if b.isChecked():
                self._chosen_layout = b.property("preset")

    def _build_page_terms(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w)
        v.addWidget(QtWidgets.QLabel("<h2>3. Termo de Consentimento e Uso</h2>"))
        browser = QtWidgets.QTextBrowser()
        browser.setOpenExternalLinks(True)
        txt = _load_terms_text()
        try:
            browser.setMarkdown(txt)
        except Exception:
            browser.setPlainText(txt)
        v.addWidget(browser, 1)
        self.chk_terms = QtWidgets.QCheckBox(
            "Li, entendi e concordo com o Termo de Consentimento e Uso do Software.")
        self.chk_terms.toggled.connect(lambda _on: self._refresh_nav())
        v.addWidget(self.chk_terms)
        return w

    def _go_back(self):
        i = self.stack.currentIndex()
        if i > 0: self.stack.setCurrentIndex(i - 1)
        self._refresh_nav()

    def _go_next(self):
        i = self.stack.currentIndex()
        if i < self.stack.count() - 1:
            self.stack.setCurrentIndex(i + 1); self._refresh_nav()
        else:
            self.accept()

    def _refresh_nav(self):
        i = self.stack.currentIndex()
        last = (i == self.stack.count() - 1)
        self.btn_back.setEnabled(i > 0)
        self.btn_next.setText("Concluir" if last else "Avançar")
        self.btn_next.setEnabled(self.chk_terms.isChecked() if last else True)

    def accept(self):
        if not self.chk_terms.isChecked():
            return  # trava defensiva
        try:
            code = self.lang_combo.currentData()
            if code: self.config.language = code
            self.config.layout_slots_cfg = [
                {"kind": k, "channel": 0}
                for k in WIZARD_LAYOUT_PRESETS.get(
                    self._chosen_layout, ["ts1", "fft", "head", "bands"])]
            self.config.terms_accepted    = True
            self.config.terms_version     = TERMS_VERSION
            self.config.terms_accepted_at = datetime.now().isoformat(timespec="seconds")
            self.config.first_run_done    = True
            self.config.save()
        except Exception:
            logging.getLogger("eeg").exception("Falha salvando aceite do termo")
        super().accept()


# ============================================================
# Catalogo de erros + notificacao com codigo (Erro E0XX)  [gerado]
# ============================================================
# Cada entrada: codigo -> (titulo, mensagem amigavel, bloqueante?)
ERROR_CATALOG = {
    'E001': ('Falha ao abrir porta COM', 'Não foi possível abrir a porta selecionada. Verifique se o dispositivo está ligado, se a porta está correta e se nenhum outro programa (OpenBCI GUI, Arduino IDE) a está usando.', True),
    'E002': ('Nenhuma porta COM selecionada', 'Nenhuma porta COM selecionada. Conecte a placa por USB e clique em Atualizar para listar as portas.', False),
    'E003': ('Baud rate inválido', 'Baud rate inválido. Escolha um valor numérico (ex.: 115200).', False),
    'E004': ('Conectado mas sem dados (watchdog)', 'Conectado a a porta selecionada, mas nenhum dado chegou em algunss. Verifique o firmware da placa, o baud rate e o cabo. Tente reconectar.', True),
    'E005': ('Perda de conexão durante aquisição', 'A conexão com a porta selecionada foi perdida durante a aquisição (dispositivo removido?). Reconecte o cabo e clique em Conectar novamente.', True),
    'E006': ('Pacotes corrompidos / dessincronização', 'Muitos pacotes corrompidos recebidos de a porta selecionada. Verifique o baud rate e a qualidade do cabo USB; os dados podem estar incompletos.', False),
    'E007': ('Comando de expansão não enviado', 'Não foi possível enviar o comando de expansão à placa. Reconecte e tente novamente; a contagem de canais pode estar inconsistente.', False),
    'E008': ('Início de streaming sem resposta', 'A placa abriu mas não respondeu ao comando de início de streaming. Verifique o firmware/baud e reconecte.', False),
    'E009': ('Daisy 16ch sem emparelhar', 'Em modo 16 canais, os pacotes da placa não estão emparelhando. A qualidade/contagem de canais pode estar incorreta — verifique se a Daisy está conectada.', False),
    'E010': ('CSV de playback não encontrado', 'Arquivo CSV de playback não encontrado. Selecione um arquivo válido em modo Playback.', True),
    'E011': ('CSV de playback malformado', 'Não foi possível ler o CSV de playback (formato inválido ou corrompido). Use um CSV gerado por este programa.', True),
    'E012': ('Nº de canais do playback adivinhado', 'Não identifiquei com certeza o número de canais do CSV; assumindo alguns. Confira o seletor de canais.', False),
    'E013': ('Biblioteca bleak ausente (BLE)', "Biblioteca 'bleak' não instalada. Instale com pip install bleak para escanear dispositivos BLE.", False),
    'E014': ('Falha no scan Bluetooth', 'Falha ao escanear Bluetooth. Verifique se o Bluetooth do Windows está ligado e se o app tem permissão.', False),
    'E015': ('Falha ao iniciar outlet LSL', 'Não foi possível iniciar o streaming LSL. Verifique se pylsl/liblsl está instalada corretamente.', False),
    'E016': ('Envio LSL falhando continuamente', 'O envio de dados via LSL está falhando — o streaming externo pode ter parado. Reinicie o LSL.', False),
    'E017': ('Autostart da simulação falhou', "Não foi possível iniciar a simulação automaticamente. Vá em Configurar → Conexão, selecione 'Simulação' e clique em Conectar.", False),
    'E018': ('Configuração do launcher não aplicada', 'Parte das configurações iniciais não pôde ser aplicada. Revise porta, canais e voluntário antes de iniciar a coleta.', False),
    'E019': ('Thread não encerrou no timeout', 'A conexão anterior demorou a encerrar. Se a próxima conexão falhar, feche e reabra o programa.', False),
    'E020': ('Exceção interna na simulação', 'Falha interna na simulação. Desconecte e tente novamente.', False),
    'E100': ('Gravação parou de salvar em disco', 'A gravação parou de salvar (disco cheio ou unidade removida). Os dados a partir de agora estão sendo PERDIDOS. Pare a gravação, libere espaço e reinicie a coleta.', True),
    'E101': ('Referência de tempo da sessão ausente', 'Referência de tempo da sessão não inicializada; timestamps inválidos. Reconecte o dispositivo antes de gravar.', False),
    'E102': ('Filtro em tempo real falhou (gravação crua)', 'O filtro digital falhou em alguns amostras; elas foram gravadas sem filtragem. A sessão será marcada como degradada — confira os parâmetros de filtro.', False),
    'E103': ('events.csv não pôde ser criado', 'Não foi possível criar events.csv; os marcadores não serão mapeados às amostras. A gravação continua, mas a análise por eventos ficará limitada.', False),
    'E104': ('Marcador não gravado em events.csv', 'Falha ao registrar um marcador em events.csv. Verifique o espaço em disco; alguns marcadores podem não constar no arquivo.', False),
    'E105': ('Trilha de auditoria não gravada', 'Não foi possível registrar a trilha de auditoria (events.jsonl). A coleta continua, mas o histórico de ações pode ficar incompleto.', False),
    'E106': ('Pasta da sessão não pôde ser criada', 'Não foi possível criar a pasta da sessão em o caminho informado (permissão ou disco). Escolha outra pasta de salvamento e tente novamente.', True),
    'E107': ('Arquivos da sessão (data.csv/log) não criados', 'Não foi possível criar o arquivo de dados da sessão. A gravação não iniciou. Verifique permissões e espaço em disco.', True),
    'E108': ('Disco insuficiente antes de gravar', 'Restam apenas poucos MB livres em a unidade. Uma sessão longa pode encher o disco e perder dados. Libere espaço antes de gravar.', False),
    'E109': ('summary.json não gravado', 'Não foi possível gravar summary.json; a sessão ficará sem metadados (canais/filtros/voluntário). A gravação continua.', False),
    'E110': ('Ficha do voluntário não anexada', 'Não foi possível anexar a ficha do voluntário à sessão. Os dados do participante podem não constar nesta pasta.', False),
    'E111': ('Assinatura de integridade (SHA-256) falhou', 'Não foi possível gerar a assinatura de integridade desta sessão. O arquivo foi salvo, mas sem prova anti-adulteração.', False),
    'E112': ('Histórico do voluntário não atualizado', 'A sessão foi salva, mas não foi possível atualizar o histórico do voluntário. Verifique a ficha do participante.', False),
    'E113': ('Falha ao finalizar/fechar arquivos', 'Houve um problema ao finalizar os arquivos da sessão. Verifique se data.csv não está aberto em outro programa antes de exportar.', False),
    'E114': ('Snapshot PNG não salvo', 'Não foi possível salvar um ou mais snapshots (disco/recurso gráfico). A coleta continua; verifique a pasta snapshots/.', False),
    'E115': ('Falha ao exportar EDF', 'Falha ao exportar EDF. Verifique se data.edf não está aberto em outro programa e tente de novo.', True),
    'E116': ('Falha ao exportar FIF', 'Falha ao exportar FIF. Verifique se o arquivo não está aberto/bloqueado e tente novamente.', True),
    'E117': ('Falha ao gerar estrutura BIDS', 'Falha ao gerar a estrutura BIDS. A pasta pode ter ficado incompleta; verifique permissões da pasta raiz e refaça a exportação.', True),
    'E118': ('Falha ao gerar relatório PDF', 'Não foi possível gerar o relatório PDF. Verifique se report.pdf não está aberto em outro programa e tente novamente.', True),
    'E200': ('data.csv corrompido/vazio ao carregar', 'Não foi possível ler os dados deste CSV — ele pode estar corrompido, vazio ou conter texto onde deveria haver números. Reexporte a sessão ou escolha outro arquivo.', True),
    'E201': ('CSV sem colunas de canal (*_uV)', "Este arquivo não tem colunas de canal reconhecíveis (ex.: 'Ch1_uV'). Confirme que é um CSV de sessão gravado pelo programa.", True),
    'E202': ('Marcadores/eventos não extraídos do CSV', 'Não foi possível extrair os marcadores/eventos do CSV. As análises baseadas em eventos ficarão indisponíveis para esta sessão.', False),
    'E203': ('Taxa de amostragem assumida (timestamps inválidos)', 'Não foi possível medir a taxa de amostragem; assumindo Hz. As análises de tempo/frequência podem ficar incorretas se a taxa real for outra.', False),
    'E204': ('SR medida diverge da declarada (>5%)', 'A taxa de amostragem medida difere >5% da declarada. As frequências/bandas podem estar incorretas; verifique a configuração de SR da gravação.', False),
    'E205': ('Amostra insuficiente para estatística (n<2)', 'Pelo menos um grupo tem menos de 2 sessões — sem dados suficientes para um teste estatístico. Adicione mais sessões.', False),
    'E206': ('Pareamento com grupos de tamanhos diferentes', "Você marcou 'amostras pareadas', mas os grupos têm tamanhos diferentes (×). Foram pareadas apenas as primeiras — confirme a correspondência ou desmarque 'pareado'.", False),
    'E207': ('Teste estatístico não pôde ser calculado', 'O teste estatístico não pôde ser calculado para esta banda (ex.: dados constantes ou variância nula). Resultado omitido.', False),
    'E208': ('Falha ao salvar relatório de estatística', "Não consegui salvar o relatório em 'o caminho informado'. Verifique espaço em disco e permissão de escrita (pastas do OneDrive em sincronização podem bloquear).", True),
    'E209': ('Treino do classificador BCI falhou', 'O treino do classificador falhou — costuma ocorrer quando os canais são muito parecidos, há poucos dados ou o sinal tem valores inválidos. Recolha mais trials ou troque os canais. (detalhe: )', True),
    'E210': ('Trial BCI capturado sem filtro (banda inválida)', 'A banda de filtragem (- Hz) é inválida (deve ser 0 < baixa < alta < Hz). O trial foi capturado sem filtro — corrija a banda antes de treinar.', False),
    'E211': ('ERD/ERS todo zerado (janelas curtas/baseline nula)', 'As janelas MI/baseline são curtas demais ou a potência de repouso é ~zero — o ERD pode estar zerado e não confiável. Use trials mais longos e verifique o contato dos eletrodos.', False),
    'E212': ('ERD% indefinido — segmentos descartados', 'Alguns segmentos foram ignorados (potência de repouso ~zero ou evento curto demais). Verifique se há sinal válido no canal/baseline escolhidos.', False),
    'E213': ('Classificação online BCI falhando', 'A classificação online está falhando repetidamente (sinal ausente ou filtro inválido). Verifique a conexão e a banda.', False),
    'E214': ('Métricas de tempo real não calculáveis (Foco/EMG/ECG/EoG)', 'Não foi possível calcular as métricas de (sinal insuficiente/inválido no canal selecionado). Verifique o canal e o filtro.', False),
    'E300': ('config.json corrompido (reset silencioso)', 'Não foi possível ler suas configurações (config.json corrompido). Voltamos ao padrão. Há uma cópia em config.json.bak — deseja restaurá-la?', False),
    'E301': ('Falha ao salvar config.json', 'Não foi possível salvar suas configurações. Verifique espaço em disco e se a pasta não está bloqueada (OneDrive/antivírus). Suas mudanças podem se perder ao fechar.', True),
    'E302': ('Pasta de salvamento padrão não criável', 'Não foi possível criar a pasta de gravação (o caminho informado). Escolha outra pasta com permissão de escrita em Configurações → Caminhos.', True),
    'E303': ('Troca de pasta de salvamento falhou', 'Não foi possível usar a pasta o caminho informado (sem permissão ou caminho inacessível). A pasta de salvamento continua a anterior. Escolha outra.', True),
    'E304': ('Tema salvo inexistente (fallback)', 'O tema salvo (o recurso) não foi encontrado — aplicamos o padrão. Reselecione ou recrie seu tema em Configurações → Tema.', False),
    'E305': ('Layout salvo com painel inexistente', 'Um ou mais painéis do seu layout salvo não existem nesta versão e foram substituídos pelo padrão. Reorganize em Visualizar → Layout Custom.', False),
    'E306': ('Salvar layout/tema reportou sucesso falso', 'A configuração foi aplicada, mas não pôde ser salva para as próximas sessões (veja o aviso sobre config.json).', False),
    'E307': ('Aceite do Termo de Uso não persistido', 'Você aceitou o termo, mas não foi possível registrar o aceite em disco. O assistente vai reaparecer na próxima abertura. Verifique espaço/permissões.', True),
    'E308': ('Termo de Uso completo ausente (resumo exibido)', 'O texto completo do Termo de Uso não foi encontrado; exibindo a versão resumida. O documento completo está em https://github.com/rodrigooa43-create/OpenBionica.', False),
    'E309': ('Configurações não salvas ao sair', 'Não foi possível salvar suas configurações ao sair. Verifique espaço/permissões. Deseja fechar mesmo assim?', True),
    'E400': ('Manifesto de update inválido/incompleto', 'O servidor respondeu, mas as informações da nova versão estão incompletas/corrompidas. Tente mais tarde; sua versão atual foi mantida.', False),
    'E401': ('SHA-256 do download não confere', 'A atualização baixada está corrompida ou adulterada e não foi aplicada. Verifique sua conexão e tente novamente; a versão atual foi preservada.', True),
    'E402': ('Update sem assinatura SHA-256', 'Esta atualização não inclui assinatura de verificação (SHA-256). Por segurança, baixe a nova versão manualmente em https://github.com/rodrigooa43-create/OpenBionica.', True),
    'E403': ('Falha ao gravar o arquivo atualizado', 'Não foi possível gravar a atualização (sem permissão na pasta do programa, ou o app está aberto em outra janela). Feche outras cópias / rode como administrador, ou baixe manualmente em https://github.com/rodrigooa43-create/OpenBionica.', True),
    'E404': ('update_config.json corrompido', 'O arquivo de configuração de atualização está corrompido. A verificação foi desativada; baixe atualizações manualmente em https://github.com/rodrigooa43-create/OpenBionica.', False),
    'E500': ('Exportação EDF indisponível (pyedflib ausente)', 'Exportação EDF indisponível: a biblioteca pyedflib não está instalada. Instale com pip install pyedflib ou use outro formato (FIF/BIDS).', True),
    'E501': ('Exportação FIF indisponível (MNE ausente)', 'Exportação FIF indisponível: a biblioteca MNE não está instalada (pip install mne). Use EDF/BIDS como alternativa.', True),
    'E502': ('Relatório PDF indisponível (reportlab/matplotlib)', 'Geração de PDF indisponível: faltam reportlab e/ou matplotlib (pip install reportlab matplotlib).', True),
    'E503': ('Dependência instalada mas não carrega (DLL/ABI)', 'O recurso o recurso está instalado mas não pôde ser carregado (instalação corrompida ou incompatível). Reinstale com pip install --force-reinstall o pacote.', True),
    'E600': ('Diretório de log não pôde ser criado', 'Não foi possível criar o arquivo de log em Documentos\\EEG_Coletor\\logs. Os relatórios de erro não serão salvos em disco. Verifique permissões/espaço.', False),
    'E601': ('Captura de tela falhou', 'Não foi possível salvar a captura de tela. Verifique espaço em disco e permissão na pasta de sessões.', False),
    'E602': ('Ação de atalho (hotkey) falhou', 'Não foi possível executar a ação do atalho.', False),
    'E603': ('Abrir pasta/caminho falhou', "Não foi possível abrir 'o item'. Confirme que o caminho existe e está acessível.", False),
    'E604': ('Logos/ícones ausentes', 'Arquivos de logo/ícone não encontrados — a interface segue funcional sem eles. Coloque os arquivos ao lado do aplicativo se quiser exibi-los.', False),
    'E999': ('Erro inesperado (excepthook)', 'Ocorreu um erro inesperado e o aplicativo pode ficar instável. Um relatório foi salvo em Documentos\\EEG_Coletor\\logs\\app.log. Reinicie o aplicativo; se persistir, envie esse arquivo ao suporte.', True),
}


def error_info(code):
    """Retorna (titulo, mensagem, bloqueante) do codigo; fallback generico."""
    return ERROR_CATALOG.get(code, ("Erro", "Ocorreu um erro nao catalogado.", True))


def notify_error(code, detail="", parent=None, exc=None, blocking=None):
    """Sinaliza um erro ao usuario como 'Erro {codigo}': loga SEMPRE e mostra
    um dialogo (bloqueante) ou deixa para a barra de status (aviso). Seguro fora
    da GUI (cai para print)."""
    title, msg, blk = error_info(code)
    if blocking is None:
        blocking = blk
    try:
        logging.getLogger("eeg").error(
            "[%s] %s%s", code, title, (" | " + str(detail)) if detail else "",
            exc_info=(exc is not None))
    except Exception:
        pass
    app = QtWidgets.QApplication.instance()
    if app is None:
        print(f"[Erro {code}] {title}: {detail}")
        return
    body = msg + (f"\n\nDetalhe técnico: {detail}" if detail else "")
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Icon.Critical if blocking
                else QtWidgets.QMessageBox.Icon.Warning)
    box.setWindowTitle(f"Erro {code}")
    box.setText(f"<b>Erro {code} \u2014 {title}</b>")
    box.setInformativeText(body)
    if exc is not None:
        try:
            box.setDetailedText("".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)))
        except Exception:
            pass
    box.exec()


class ErrorDiagnosticsDialog(QtWidgets.QDialog):
    """Lista o catalogo de erros e permite SIMULAR cada notificacao (Erro E0XX),
    para o usuario conhecer/testar as mensagens que o programa pode exibir."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Diagnóstico de erros — simular notificações")
        self.setMinimumSize(780, 580)
        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(QtWidgets.QLabel(
            "<b>Catálogo de erros do programa.</b> Selecione um código e clique em "
            "<b>Simular</b> para ver a notificação. No uso real, <i>avisos</i> "
            "aparecem na barra de status (rodapé) e <i>bloqueantes</i> abrem um "
            "diálogo; aqui mostramos o diálogo para você conhecer cada um."))
        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Filtrar por código ou texto…")
        self.filter_edit.textChanged.connect(self._refilter)
        v.addWidget(self.filter_edit)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Código", "Tipo", "Título"])
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.table, 1)
        self._rows = sorted(ERROR_CATALOG.items())
        self._fill(self._rows)
        self.table.itemSelectionChanged.connect(self._preview)
        self.table.doubleClicked.connect(self._simulate)
        self.preview = QtWidgets.QPlainTextEdit(); self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(120)
        v.addWidget(self.preview)
        h = QtWidgets.QHBoxLayout()
        self.count_lbl = QtWidgets.QLabel(f"{len(self._rows)} códigos catalogados")
        btn_sim = QtWidgets.QPushButton("▶ Simular notificação")
        btn_sim.clicked.connect(self._simulate)
        btn_close = QtWidgets.QPushButton("Fechar"); btn_close.clicked.connect(self.accept)
        h.addWidget(self.count_lbl); h.addStretch(1)
        h.addWidget(btn_sim); h.addWidget(btn_close)
        v.addLayout(h)
        try: self.setStyleSheet(build_stylesheet(COLORS))
        except Exception: pass
        if self.table.rowCount(): self.table.selectRow(0)

    def _fill(self, rows):
        self.table.setRowCount(len(rows))
        for r, (code, (title, _msg, blk)) in enumerate(rows):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(code))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(
                "Bloqueante" if blk else "Aviso"))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(title))
        self.table.resizeColumnToContents(0)
        self.table.resizeColumnToContents(1)

    def _refilter(self, text):
        t = (text or "").lower().strip()
        rows = [it for it in self._rows if (t in it[0].lower()
                or t in it[1][0].lower() or t in it[1][1].lower())]
        self._fill(rows)
        self.count_lbl.setText(f"{len(rows)} de {len(self._rows)} códigos")
        if self.table.rowCount(): self.table.selectRow(0)

    def _current_code(self):
        r = self.table.currentRow()
        it = self.table.item(r, 0) if r >= 0 else None
        return it.text() if it else None

    def _preview(self):
        code = self._current_code()
        if not code: return
        title, msg, blk = error_info(code)
        self.preview.setPlainText(
            f"Erro {code} — {title}  [{'Bloqueante' if blk else 'Aviso'}]\n\n{msg}")

    def _simulate(self, *_):
        code = self._current_code()
        if not code:
            QtWidgets.QMessageBox.information(self, "Simular",
                                              "Selecione um código na lista.")
            return
        notify_error(code, "(simulação)", parent=self, blocking=True)


# ============================================================
# Assistente de Ajuda (OFFLINE) — busca local em FAQ + erros + manual
# ============================================================
HELP_FAQ = [
    {"t": "Conectar ao dispositivo",
     "k": "como conectar dispositivo placa openbci cyton porta com iniciar coleta hardware",
     "a": "Vá em <b>Configurar → Conexão</b>, escolha a <b>Porta</b> COM, o nº de canais "
          "e o modo, e clique em <b>Conectar</b>. Sem hardware, use o <b>Modo "
          "Simulação</b> na tela inicial.",
     "ref": "Manual: Coleta → Conexão"},
    {"t": "Nenhuma porta COM aparece",
     "k": "porta com nao aparece vazia lista nao encontra dispositivo usb cabo",
     "a": "Verifique o cabo/dongle, clique em <b>Atualizar</b> na aba Conexão e feche "
          "outros programas que usem a porta (OpenBCI GUI, Arduino IDE). Para testar "
          "sem hardware, use o <b>Modo Simulação</b>.",
     "ref": "Erro E001/E002"},
    {"t": "Calibração e impedância",
     "k": "impedancia calibracao contato eletrodo verde vermelho qualidade sinal ruido",
     "a": "Em <b>Configurar → Calibração</b>, rode o teste de impedância. Valores "
          "baixos (verde) = bom contato. O indicador no topo mostra ● OK / ▲ ruidoso "
          "/ ■ ruim por canal.",
     "ref": "Manual: Calibração"},
    {"t": "Filtros (notch e passa-banda)",
     "k": "filtro notch 60 50 hz passa banda bandpass ruido rede eletrica",
     "a": "Em <b>Configurar → Filtros e Canais</b>, ative o <b>notch</b> (60 Hz no "
          "Brasil) e ajuste o <b>passa-banda</b> conforme a análise (ex.: 1–40 Hz "
          "para EEG).",
     "ref": "Manual: Filtros e canais"},
    {"t": "Gravar uma sessão",
     "k": "gravar sessao iniciar parar registro salvar coleta botao",
     "a": "Conecte o dispositivo e clique em <b>Gravar</b> (ou Ctrl+R). Os dados vão "
          "para uma pasta por sessão (data.csv, summary.json, …). Clique de novo para "
          "parar.",
     "ref": "Manual: Gravar uma sessão"},
    {"t": "Onde ficam minhas gravações",
     "k": "onde ficam gravacoes arquivos sessoes pasta salvar local csv encontrar",
     "a": "Na subpasta <b>sessions/</b> (ou na pasta de salvamento configurada). "
          "Atalho: <b>Ajuda → Pasta de configuração / sessões</b>.",
     "ref": "Manual: Solução de problemas"},
    {"t": "Marcadores de evento",
     "k": "marcador evento marker hotkey tecla m estimulo anotar gatilho",
     "a": "Durante a gravação, registre um marcador pela tecla de atalho (ex.: <b>M</b>) "
          "ou pelo painel de eventos. Eles ficam em events.csv e servem à análise por "
          "eventos.",
     "ref": "Manual: Coleta"},
    {"t": "Análises (FFT, bandas)",
     "k": "analise fft espectro bandas delta teta alfa beta gama potencia frequencia",
     "a": "Em <b>Analisar → Análises</b> você vê a FFT, a potência por banda "
          "(δ θ α β γ) e estatísticas por canal da janela selecionada.",
     "ref": "Manual: Análises"},
    {"t": "Topografia e espectrograma",
     "k": "topografia mapa calor head plot espectrograma tempo frequencia",
     "a": "Em <b>Visualizar → Topografia</b> veja o mapa de calor (10–20); em "
          "<b>Espectrograma</b>, a frequência ao longo do tempo.",
     "ref": "Manual: Topografia / Espectrograma"},
    {"t": "ERS/ERD (imagética motora)",
     "k": "ers erd imagetica motora dessincronizacao mu beta baseline repouso",
     "a": "Em <b>Analisar → ERS/ERD</b>, carregue a sessão, escolha a classe, a banda "
          "(ex.: Mu 8–13 Hz) e a baseline (repouso) e clique em Computar. ERD < 0 "
          "indica dessincronização.",
     "ref": "Manual: ERS/ERD"},
    {"t": "Comparar Antes × Depois (estatística)",
     "k": "comparar grupos antes depois estatistica teste significancia p valor exo",
     "a": "Em <b>Analisar → Offline</b>, use <b>Estatística guiada (comparar grupos)</b>: "
          "selecione as sessões de cada grupo; o programa escolhe o teste sozinho, "
          "monta a tabela por banda e explica o resultado.",
     "ref": "Manual: Facilitador estatístico"},
    {"t": "Comparar condições de uma sessão",
     "k": "comparar condicoes sessao classes repouso metrica banda rms erd intra",
     "a": "Em <b>Analisar → Offline</b>, use <b>Comparar condições da sessão</b>: compara "
          "as condições que existirem (classes, tarefa × repouso) pela métrica escolhida "
          "(banda, RMS ou ERD%).",
     "ref": "Manual: Facilitador estatístico"},
    {"t": "Área Maker — Receitas de análise",
     "k": "receita area maker pipeline montar analise template reutilizar salvar json composavel",
     "a": "Em <b>Analisar → Offline → Receitas de análise (Área Maker)</b> você monta "
          "uma análise (métrica + banda + canais), roda em uma ou várias sessões e "
          "<b>salva como .json</b> para reutilizar/compartilhar o protocolo.",
     "ref": "Área Maker"},
    {"t": "Exportar EDF / FIF / BIDS / PDF",
     "k": "exportar edf fif bids pdf relatorio formato clinico cientifico converter salvar",
     "a": "Em <b>Configurar → Sessão e Arquivos</b>, exporte para EDF/EDF+, FIF (MNE), "
          "BIDS ou um relatório PDF. (FIF exige rodar via Python; no .exe use EDF/BIDS.) "
          "Se falhar, veja os Erros E115–E118.",
     "ref": "Manual: Exportação"},
    {"t": "Atualizações (offline/manual)",
     "k": "atualizar atualizacao versao nova update offline github verificar internet",
     "a": "O programa é <b>offline por padrão</b>. Para checar, use <b>Ajuda → Verificar "
          "atualizações</b>. Ele só baixa código (nunca envia dados) e confere a "
          "assinatura SHA-256.",
     "ref": "Manual: Atualizações"},
    {"t": "Privacidade e Termo de Uso",
     "k": "privacidade dados lgpd termo offline coleta envia compartilha seguranca",
     "a": "O software funciona <b>offline</b> e <b>não coleta, não envia e não "
          "compartilha</b> dados — tudo fica na sua máquina. Releia em <b>Ajuda → Termo "
          "de uso e privacidade</b>.",
     "ref": "Termo de Uso"},
    {"t": "Códigos de erro (Erro E0XX)",
     "k": "erro codigo problema falha mensagem o que significa resolver",
     "a": "Quando algo falha, aparece um <b>código</b> (ex.: Erro E115). Digite o código "
          "aqui para a explicação, ou abra <b>Ajuda → Diagnóstico de erros</b> para a "
          "lista completa.",
     "ref": "ERROS.md"},
    {"t": "Treinador BCI",
     "k": "bci treinador csp lda classificador imagetica treinar acuracia",
     "a": "Em <b>Analisar → BCI Trainer</b>, treine um classificador (CSP + LDA) a partir "
          "dos trials gravados; o programa mostra a acurácia.",
     "ref": "Manual: Treinador BCI"},
    {"t": "Onde fica o log de erros",
     "k": "log arquivo erro relatorio suporte diagnostico applog problema",
     "a": "Em <b>Documentos/EEG_Coletor/logs/app.log</b>. Anexe esse arquivo ao relatar "
          "um problema — ajuda a diagnosticar.",
     "ref": "Manual: Solução de problemas"},
    {"t": "Abrir arquivo EDF / BDF",
     "k": "abrir edf bdf importar ler arquivo edfbrowser icelera converter reparo "
          "ascii cabecalho quebrado nao abre erro caractere",
     "a": "Em <b>Analisar → Offline</b> clique em <b>Abrir EDF/BDF (reparo "
          "automático)</b>. O leitor é <b>tolerante</b>: ignora cabeçalhos com "
          "caracteres inválidos (acentos no nome do paciente) que fazem o EDFbrowser/"
          "pyedflib recusarem o arquivo. Ele converte para CSV nativo (respeitando a "
          "taxa de amostragem real) e já abre para análise. Também funciona ao "
          "cadastrar um voluntário e importar o exame.",
     "ref": "Manual: Análise offline"},
    {"t": "Limpar artefatos (ICA)",
     "k": "ica artefato piscada ocular eog limpar componente independente frontal "
          "fastica mne instalar pip automatico data_clean",
     "a": "Em <b>Analisar → Offline</b>, abra a sessão e clique em <b>Limpar "
          "artefatos (ICA)</b>. Roda <b>FastICA em numpy puro</b> — <b>não precisa "
          "instalar nada</b> (nem MNE). Ele filtra 1–40 Hz, detecta o componente de "
          "piscada pelos canais frontais (Fp1/Fp2) e salva <b>data_clean.csv</b> na "
          "pasta da sessão.",
     "ref": "Manual: Análise offline"},
    {"t": "Topografia CSD / Laplaciano (vs LORETA)",
     "k": "topografia csd laplaciano superficie current source density conducao "
          "volume loreta sloreta fonte localizacao mne eeglab mapa nitido fonte local",
     "a": "Na aba <b>Topografia</b>, troque <b>Mapa</b> para <b>CSD / Laplaciano "
          "(fonte)</b>: realça fontes locais e reduz a condução de volume (mapa mais "
          "nítido que o interpolado). O botão <b>vs LORETA / MNE…</b> explica como "
          "isso se compara à localização de fonte 3D (LORETA/sLORETA) e quando "
          "exportar para MNE/EEGLAB.",
     "ref": "Manual: Topografia"},
    {"t": "Área Maker — ERD% e modelos prontos",
     "k": "area maker receita erd baseline potencia relativa relband modelo pronto "
          "template cenario openvibe metrica mu beta imagetica pipeline",
     "a": "A <b>Área Maker</b> agora tem as métricas <b>ERD% vs baseline</b> "
          "(imagética motora) e <b>potência relativa da banda (%)</b>, além de "
          "<b>Modelos prontos</b> (ex.: 'ERD Mu — imagética motora', 'Alpha relativo') "
          "que preenchem a receita com valores embasados. Para o ERD, defina a "
          "<b>baseline</b> (janela de repouso) e o <b>recorte</b> (janela da tarefa).",
     "ref": "Manual: Área Maker"},
    {"t": "Como o Consultor pensa (metodologia)",
     "k": "consultor assistente metodologia abordar estudo raciocinio focado "
          "referencias literatura embasado como devo aprofundar sugestao",
     "a": "O <b>Consultor</b> (F1) tenta entender sua <b>intenção</b> e dá uma "
          "resposta <b>focada e com referências</b> — não despeja texto genérico. "
          "Pergunte de metodologia (ex.: <i>como abordo EEG de membro superior?</i>, "
          "<i>qual banda/janela para MI?</i>, <i>como evitar vazamento na validação?</i>) "
          "e ele responde com base em literatura e sugere <b>o que fazer no programa</b> "
          "e <b>perguntas de aprofundamento</b>. Se não souber, ele diz e orienta "
          "contatar o suporte ou contribuir no código aberto.",
     "ref": "Consultor OpenBionica"},
]


def _help_norm(s):
    s = unicodedata.normalize("NFKD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", s)


_HELP_STOP = {"como", "fazer", "faco", "faz", "quero", "qual", "quais", "onde",
              "meu", "minha", "meus", "minhas", "para", "por", "que", "isso",
              "esse", "essa", "com", "sem", "uma", "dos", "das", "nao", "sim",
              "tem", "ter", "ver", "aqui", "sobre", "esta", "este", "preciso",
              "gostaria", "posso", "pode", "the", "and", "you", "meu"}


def _help_tokens(s):
    return set(t for t in _help_norm(s).split()
               if len(t) > 2 and t not in _HELP_STOP)


def _help_strip_html(s):
    return re.sub(r"<[^>]+>", " ", s or "")


def _help_kb_to_html(text):
    """Converte uma passagem da base de conhecimento (texto) em HTML curto."""
    lines = [l.strip() for l in _help_strip_html(text).split("\n") if l.strip()]
    html = []
    for l in lines[:9]:
        html.append(("• " + l[2:]) if l.startswith("- ") else l)
    return "<br>".join(html)


# ============================================================
# Bases de conhecimento EMBUTIDAS (fallback p/ atualizacao a quente,
# que troca so o .py). Se o .md existir em disco, ele tem prioridade.
# ============================================================
GUIA_METODOLOGICO_EMBED = """
# Guia metodológico — EEG, Imaginação Motora e BCI

Base de conhecimento consultiva do OpenBionica. As orientações abaixo são
**embasadas em literatura publicada** (referências ao final de cada tema). O
assistente usa este guia para responder "como devo abordar…", "qual a melhor
prática para…" e sugerir caminhos coerentes. Quando um tema não estiver aqui,
o assistente recomenda contatar o suporte/orientador ou contribuir no código
aberto.

---

## Como abordar um estudo de imaginação motora de membro superior

Roteiro prático e embasado, do zero ao resultado:

1. **Pergunta e desenho.** Defina as classes (ex.: mão esquerda × direita;
   ou mão × repouso). Para pós-AVC e reabilitação, foque em **imaginação
   cinestésica** (sentir o movimento), não visual — ela gera ERD/ERS mais
   robustos [Neuper 2005].
2. **Montagem.** Padrão internacional 10–20, priorizando o córtex
   sensoriomotor: **C3, Cz, C4** (mão/membro superior) e, se possível,
   FC3/FC4, CP3/CP4 para melhorar filtros espaciais [Klem 1999; Blankertz 2008].
3. **Paradigma por trial.** Repouso → cue (aviso) → janela de MI (~4–5 s) →
   intervalo. Descarte os primeiros ~0,5 s (reação) e analise **0,5–2,5 s
   pós-cue**, janela consagrada na BCI Competition IV [Ang 2008; Tangermann 2012].
4. **Bandas.** **mu (8–13 Hz)** e **beta (13–30 Hz)** são as bandas
   sensoriomotoras onde ocorre a dessincronização (ERD) durante a MI e a
   ressincronização (ERS / beta rebound) após [Pfurtscheller & Lopes da Silva 1999].
5. **Pré-processamento.** Filtro passa-banda (ex.: 8–30 Hz para MI), notch da
   rede (50/60 Hz), remoção de artefatos oculares por regressão ou ICA
   [Makeig 1996; Jung 2000].
6. **Extração de padrões.** ERD/ERS por banda e **filtros espaciais** —
   Laplaciano de superfície (simples) ou **CSP/FBCSP** (mais potente para
   classificação) [Ramoser 2000; Blankertz 2008; Ang 2008].
7. **Classificação e validação.** LDA/CSP é a linha de base robusta; valide com
   **validação cruzada agrupada por trial** (GroupKFold) para **não haver
   vazamento** entre janelas do mesmo trial [Lotte 2018; Brookshire 2024].
8. **Amostra.** Espere que 15–30% dos voluntários sejam "BCI-illiterate"
   (baixo desempenho) — faça triagem (ex.: MIQ-RS) e planeje n com folga
   [Sannelli 2019]. Colete em **≥ 2 dias** para capturar variabilidade
   inter-dia [Ma 2022].

**No OpenBionica:** grave em *Nova Coleta*/protocolo → analise em
*Analisar → Offline* (FFT, bandas, ERS/ERD, topografia) → use a *Área Maker*
para montar a métrica (recorte → filtro → banda) e rodar em várias sessões.

> Refs: Pfurtscheller & Lopes da Silva 1999; Neuper 2005; Klem 1999;
> Blankertz 2008; Ang 2008; Ramoser 2000; Lotte 2018; Ma 2022; Sannelli 2019.

---

## ERD/ERS — o que é e como calcular

**ERD (Event-Related Desynchronization)** é a queda de potência em mu/beta
sobre o córtex sensoriomotor durante o preparo/execução/imaginação do
movimento; **ERS** é o aumento (ressincronização), típico do *beta rebound*
pós-movimento. É o marcador fisiológico central da imaginação motora
[Pfurtscheller & Lopes da Silva 1999].

**Cálculo (método clássico, "band power"):**
1. Filtre na banda de interesse (ex.: mu 8–13 Hz).
2. Eleve o sinal ao quadrado (potência instantânea) e suavize (média móvel).
3. ERD%(t) = 100 × (P(t) − P_ref) / P_ref, onde **P_ref** é a potência média
   em uma **janela de referência (baseline)** de repouso antes do cue.
- ERD% negativo = dessincronização (esperado na MI); positivo = ERS.
- Lateralização: compare **C3 × C4** (a MI de uma mão gera ERD contralateral).

**Boas práticas:** baseline de 1–2 s imediatamente antes do cue; média sobre
trials da mesma classe; reporte por banda e por canal.

> Refs: Pfurtscheller & Lopes da Silva 1999; Pfurtscheller & Neuper 2001.

---

## Cinestésica vs visual, e quantos trials

**Imaginação cinestésica** (sentir a contração/esforço, 1ª pessoa) ativa mais
M1/SMA e produz ERD/ERS mais classificáveis do que a **visual** (ver-se de
fora) [Neuper 2005]. Instrua o voluntário a *sentir* o movimento sustentado,
sem executar de verdade.

**Número de trials:** para análise/treino robusto, mire **≥ 40–72 trials por
classe** distribuídos em partes com pausas (fadiga cai o desempenho). Para BCI
online, calibração de ~20–40 trials/classe é comum; datasets canônicos usam
72/classe (BCI Comp IV-2a) [Tangermann 2012].

> Refs: Neuper 2005; Tangermann 2012.

---

## Artefatos: como limpar (piscadas, músculo, rede)

- **Piscadas/olhos (EOG):** aparecem forte em Fp1/Fp2. Remova por **ICA**
  (excluir componentes de blink) ou regressão por canais frontais
  [Makeig 1996; Jung 2000]. No OpenBionica: *Analisar → Offline →
  "Limpar artefatos (ICA)"* — roda em numpy puro, **sem instalar nada**, e
  salva `data_clean.csv`.
- **Músculo (EMG):** energia em alta frequência (>30 Hz), pior em T3/T4.
  Evite instruindo relaxamento; filtre e/ou marque trials ruins.
- **Rede elétrica:** notch em 50 ou 60 Hz (Brasil = 60 Hz).
- **Regra prática:** trial com artefato deve ser **sinalizado e excluído** do
  treino, não "corrigido à força".

> Refs: Makeig 1996; Jung 2000; Picton 2000; Keil 2014.

---

## Classificação e validação (evitar vazamento)

- **Linha de base robusta:** CSP (filtro espacial) + LDA. FBCSP para
  multi-banda [Ramoser 2000; Ang 2008]. Revisão ampla de classificadores em
  [Lotte 2018].
- **Validação:** nunca misture janelas do **mesmo trial** entre treino e teste.
  Use **GroupKFold por trial_id**; para generalização entre pessoas, use
  **LOSO (Leave-One-Subject-Out)**. Ignorar isso infla a acurácia
  artificialmente [Brookshire 2024].
- **Cross-session:** desempenho cai muito entre dias sem adaptação; treine com
  dados de ≥ 2 dias [Ma 2022; Huang 2023].
- **Métricas:** reporte acurácia **e** kappa de Cohen (corrige acaso).

> Refs: Ramoser 2000; Ang 2008; Lotte 2018; Brookshire 2024; Ma 2022.

---

## Análise visual: topografia, LORETA e localização de fonte

**Mapa topográfico (topomap)** projeta a potência/banda por eletrodo em uma
vista da cabeça (interpolação 2D). É rápido e ótimo para ver *onde* está a
atividade, mas mostra **potencial de escalpo**, não a fonte — sofre de
**condução de volume** [Nunez & Srinivasan 2006].

Para ir além do topomap simples:
- **Laplaciano de superfície / CSD (Current Source Density):** realça fontes
  locais e reduz condução de volume; é uma referência espacial "sem
  referência" [McFarland 1997; Nunez & Srinivasan 2006]. É barato (só a
  geometria dos eletrodos) e melhora muito a nitidez do topomap.
- **Localização de fonte (LORETA/sLORETA):** estima a distribuição de corrente
  no volume cerebral a partir do EEG de escalpo. **LORETA** assume solução
  suave (mínima Laplaciana); **sLORETA** é a versão padronizada com
  localização de erro zero para fonte única [Pascual-Marqui 1994; 2002].
  Requer um modelo de cabeça/leadfield.

**Comparação honesta OpenBionica × LORETA/EEGLAB/MNE:** o OpenBionica é um
coletor+analisador leve (topomap 2D, FFT, bandas, ERS/ERD). LORETA/sLORETA e
localização de fonte 3D são o domínio de **MNE-Python** [Gramfort 2013] e do
**EEGLAB** [Delorme 2004]. Caminho recomendado: usar o OpenBionica para
coleta/triagem visual e **exportar para MNE/EEGLAB** (o software já exporta
EDF/BIDS) quando precisar de tomografia de fonte.

> Refs: Nunez & Srinivasan 2006; McFarland 1997; Pascual-Marqui 1994, 2002;
> Gramfort 2013 (MNE); Delorme 2004 (EEGLAB).

---

## Referências

- Ang KK, Chin ZY, Zhang H, Guan C (2008). *Filter Bank Common Spatial Pattern
  (FBCSP) in Brain–Computer Interface.* IJCNN, p.2390–2397.
- Blankertz B, Tomioka R, Lemm S, Kawanabe M, Müller KR (2008). *Optimizing
  spatial filters for robust EEG single-trial analysis.* IEEE Signal Process
  Mag 25(1):41–56.
- Brookshire G et al. (2024). *Data leakage in deep learning studies of
  translational EEG.* Front Neurosci 18:1373515.
- Delorme A, Makeig S (2004). *EEGLAB: an open source toolbox…* J Neurosci
  Methods 134(1):9–21.
- Gramfort A et al. (2013). *MEG and EEG data analysis with MNE-Python.* Front
  Neurosci 7:267.
- Huang G et al. (2023). *Discrepancy between inter- and intra-subject
  variability in EEG-based MI BCI.* Front Neurosci 17:1122661.
- Jung TP et al. (2000). *Removing electroencephalographic artifacts by blind
  source separation.* Psychophysiology 37(2):163–178.
- Keil A et al. (2014). *Committee report: publication guidelines… EEG/MEG.*
  Psychophysiology 51(1):1–21.
- Klem GH, Lüders HO, Jasper HH, Elger C (1999). *The ten–twenty electrode
  system of the International Federation.* Electroencephalogr Clin Neurophysiol
  Suppl 52:3–6.
- Lotte F et al. (2018). *A review of classification algorithms for EEG-based
  BCI: a 10-year update.* J Neural Eng 15(3):031005.
- Ma J et al. (2022). *A large EEG dataset for studying cross-session
  variability in MI BCI.* Sci Data 9:531.
- Makeig S, Bell AJ, Jung TP, Sejnowski TJ (1996). *Independent component
  analysis of electroencephalographic data.* NIPS 8:145–151.
- McFarland DJ, McCane LM, David SV, Wolpaw JR (1997). *Spatial filter
  selection for EEG-based communication.* Electroencephalogr Clin Neurophysiol
  103(3):386–394.
- Neuper C, Scherer R, Reiner M, Pfurtscheller G (2005). *Imagery of motor
  actions: differential effects of kinesthetic and visual-motor mode of imagery
  in single-trial EEG.* Cogn Brain Res 25(3):668–677.
- Nunez PL, Srinivasan R (2006). *Electric Fields of the Brain: The Neurophysics
  of EEG.* 2ª ed., Oxford Univ. Press.
- Pascual-Marqui RD, Michel CM, Lehmann D (1994). *Low resolution
  electromagnetic tomography (LORETA).* Int J Psychophysiol 18(1):49–65.
- Pascual-Marqui RD (2002). *Standardized low-resolution brain electromagnetic
  tomography (sLORETA).* Methods Find Exp Clin Pharmacol 24 Suppl D:5–12.
- Pfurtscheller G, Lopes da Silva FH (1999). *Event-related EEG/MEG
  synchronization and desynchronization: basic principles.* Clin Neurophysiol
  110(11):1842–1857.
- Pfurtscheller G, Neuper C (2001). *Motor imagery and direct brain–computer
  communication.* Proc IEEE 89(7):1123–1134.
- Picton TW et al. (2000). *Guidelines for using human event-related
  potentials…* Psychophysiology 37(2):127–152.
- Ramoser H, Müller-Gerking J, Pfurtscheller G (2000). *Optimal spatial
  filtering of single trial EEG during imagined hand movement.* IEEE Trans
  Rehabil Eng 8(4):441–446.
- Sannelli C, Vidaurre C, Müller KR, Blankertz B (2019). *A large scale
  screening study with a SMR-based BCI.* PLOS ONE 14(1):e0207351.
- Tangermann M et al. (2012). *Review of the BCI Competition IV.* Front Neurosci
  6:55.
"""

BASE_CONHECIMENTO_EMBED = """
# Base de Conhecimento — OpenBiônica (OpenBionica)

_Gerado automaticamente a partir do manual do usuário. É o corpus que o assistente de ajuda offline indexa para responder às perguntas._

## Introdução

O OpenBiônica é um aplicativo para coleta, visualização e análise
de sinais de eletroencefalografia (EEG) em tempo real, compatível com a placa
OpenBCI Cyton (8 ou 16 canais, 250,Hz). Ele oferece visualização multicanal,
filtros, mapas topográficos, análise espectral, ERS/ERD e um treinador de
Brain–Computer Interface (BCI), além de exportação para formatos clínicos
e científicos.

Além do EEG, o software é multimodal (aceita também EMG, ECG, EoG e
acelerômetro) e traz um facilitador estatístico que escolhe e executa o
teste adequado automaticamente — você compara condições ou grupos sem precisar
dominar estatística (Seção). Para confiabilidade, registra
proveniência (versão + repositório) em cada exportação, mantém um log de
aplicação e possui recuperação de erros que evita fechamentos silenciosos.

> Dica: 
Este é um software de pesquisa, sem aprovação como dispositivo médico.
Destina-se a pesquisa acadêmica, educação e desenvolvimento de protocolos —
não deve ser usado para diagnóstico ou decisões clínicas.

## Instalação e primeira execução

- Descompacte o arquivo Software EEG.zip em uma pasta de sua
preferência (ex.: Área de Trabalho).
- Dê um duplo clique em EEG_Collector.exe.
- Na primeira vez, o Windows pode exibir um aviso do SmartScreen.
Clique em Mais informações → Executar assim mesmo
(isso ocorre porque o executável não tem assinatura digital paga; é seguro).
- A janela abre em cerca de 10–20,s na primeira execução.

> Dica: 
O .exe já contém tudo o que é necessário para rodar. Alternativamente,
quem tiver Python 3.10+ pode executar python EEG_Data_Collector.py.

## Primeiro uso: idioma, layout e Termo de Uso

Na primeira vez que você abre o programa, um assistente guia três
passos rápidos (s e ):

- Idioma da interface (Português, English ou Español).
- Layout inicial dos painéis (pode mudar depois em
Visualizar → Layout Custom).
- Termo de Consentimento e Uso — leia e marque Li, entendi e
concordo para concluir. O programa só abre após o aceite.

> Dica: 
O software funciona offline e não coleta, não envia e não
compartilha nenhum dado — tudo permanece na sua máquina. Você pode reler o termo
a qualquer momento em Ajuda → Termo de uso e privacidade; o
aceite (versão e data) fica registrado apenas no seu computador.

## Tela inicial

Ao abrir, surge a tela de boas-vindas (, dividida
em três áreas:

- Esquerda — Voluntário e sessões: selecione um voluntário já
cadastrado ou clique em + Novo para cadastrar. Veja também as
sessões recentes e o botão Abrir CSV manualmente.
- Centro — Fluxos de trabalho: escolha o que deseja fazer:
Nova Coleta (conectar e gravar), Analisar Dados (abrir um CSV
já gravado), Aplicações BCI ou Modo Simulação (gera sinal
sintético, sem hardware).
- Direita — Pré-flight check: defina a Porta COM, o
Número de canais e o Tipo de Aquisição antes de iniciar.

> Dica: 
Para testar o programa sem hardware, escolha Modo Simulação — todas
as telas funcionam com um sinal sintético.

## Coleta de dados em tempo real

## Conexão

Na aba Configurar → Conexão (:

- Selecione a Porta COM da Cyton (use Atualizar para
relistar). Para testes, troque o Modo para Simulação.
- Confira o Baud Rate (padrão 115200).
- Dispositivos Bluetooth podem ser pareados pela seção correspondente.
- O botão Demo 30s grava automaticamente uma sessão de exemplo
(útil para validar a cadeia de processamento).

## Visualização em tempo real

A aba Visualizar → Tempo Real (
mostra todos os canais empilhados, com o acelerômetro (X/Y/Z) na parte inferior.
Ajuste a Escala ($$V/canal) e a Janela (segundos) conforme
necessário.

## Filtros e canais

Em Configurar → Filtros e Canais (
ative o notch (50/60,Hz, contra ruído de rede), o passa-banda
(ex.: 1–50,Hz) e habilite/desabilite canais individualmente.

> Dica: 
Para imagética motora, um passa-banda de 8–30,Hz (faixas mu e beta) e o
notch de 60,Hz costumam dar o melhor sinal.

## Calibração (impedância)

A aba Calibração ( realiza o teste de
impedância dos eletrodos (em k$$). Verifique se os eletrodos estão
com bom contato antes de gravar — valores altos indicam contato ruim.

## Gravar uma sessão

- Clique em Conectar e aguarde o sinal estabilizar.
- Clique em Iniciar Gravação.
- Use os Marcadores (atalhos de teclado / botões de evento) para
anotar os momentos relevantes (repouso, estímulo, imagética, etc.).
- Clique em Parar Gravação. Os arquivos são salvos em
sessions/<sujeito>_<data>_<hora>/ (EEG em data.csv,
eventos, log, summary.json com integridade SHA-256 e snapshots).

## Análise dos sinais

## Análises (FFT, bandas, estatísticas)

A aba Analisar → Análises (
exibe a FFT do canal escolhido, a potência por banda EEG ($,,,
,$) e estatísticas por canal.

## Topografia (mapa de calor)

A aba Topografia ( mostra o
head plot 10–20 com interpolação, além de Focus e EMG.

## Espectrograma

A aba Espectrograma ( apresenta o
mapa de calor frequência $×$ tempo por canal.

## Análise offline de sessões gravadas

A aba Analisar → Offline (
permite abrir um data.csv já gravado, arrastar uma região de
interesse sobre o sinal e ver, para a seleção, a FFT, as bandas e as estatísticas
por canal — respeitando a taxa de amostragem real do arquivo.

## ERS/ERD (imagética motora)

A aba Analisar → ERS/ERD (
calcula a dessincronização/sincronização relacionada a evento:

- Clique em Carregar CSV BCI e selecione a sessão.
- Escolha a Classe, a Banda (ex.: Mu 8–13,Hz) e a
Baseline (use baseline = repouso para o ERD fisiológico).
- Clique em Computar ERS/ERD. Veja a topografia, as barras de ERD%
por canal e o curso temporal. ERD,$<$,0 = dessincronização
(marcador típico de imagética motora).

## Treinador BCI

A aba Analisar → BCI Trainer (
treina um classificador (CSP + LDA) de imagética motora a partir dos trials
gravados, indicando a acurácia obtida.

## Facilitador estatístico

O software escolhe o teste estatístico por você: verifica a normalidade
(Shapiro–Wilk) e, conforme o caso, aplica t de Welch/Student, Wilcoxon,
Mann–Whitney, ANOVA ou Kruskal–Wallis; calcula o tamanho de efeito (Cohen d),
corrige múltiplas comparações (Holm) e explica o resultado em linguagem
simples. Ideal para clínicos, estudantes e engenheiros — sem exigir conhecimento
prévio de estatística.

> Dica: 
O facilitador compara as condições que existirem nos seus dados (classes,
marcadores, repouso) pela métrica que você escolher — não assume nenhum movimento
específico. Serve igualmente para tornozelo, mão, EMG, etc.

## Comparar grupos de sessões (ex.: Antes $×$ Depois)

Em Analisar → Offline, clique em Estatística guiada
(comparar grupos) (:

- Em Grupo A e Grupo B, clique em Selecionar
sessões… e escolha os arquivos de cada condição (ex.: antes e depois de
um protocolo). Marque Amostras pareadas se forem os mesmos sujeitos.
- Clique em Comparar. A tabela mostra, por banda
((,,,,)), o teste escolhido, o valor de
p, o p corrigido (Holm), o efeito e se houve diferença.
- Leia o resumo em texto e, se quiser, clique em Salvar relatório
(HTML + CSV).

## Comparar condições de uma sessão

Ainda na aba Offline, Comparar condições da sessão
( compara, dentro de uma sessão,
as condições detectadas (classes, marcadores ou tarefa $×$ repouso). Escolha
a métrica (potência de banda, RMS ou ERD% vs. repouso), a banda
e os canais; o teste é selecionado automaticamente.

## Área Maker — Receitas de análise

A Área Maker ( deixa você montar
uma análise conforme a sua necessidade, sem programar, e salvá-la como
receita para reutilizar ou compartilhar. Acesse em Analisar
→ Offline → Receitas de análise (Área Maker):

- Escolha a métrica (potência de banda, RMS, amplitude
pico-a-pico ou desvio-padrão), a banda e os canais
(vazio = todos, ou ex.: 1,2,3).
- (Opcional, pipeline) defina um recorte temporal
(início--fim em segundos) e/ou um pré-filtro passa-banda
(Hz) — a receita aplica recorte → filtro
→ métrica.
- Clique em Adicionar sessões e selecione um ou mais arquivos.
- Clique em Rodar: a tabela mostra o valor por canal, com uma
coluna por sessão (ótimo para comparar, ex.: Antes (×) Depois).
- Salvar receita grava um .json reutilizável; Carregar
receita reaplica; Exportar tabela salva o resultado em CSV.

.

## Exportação

Pela aba de Configurações → Sessão & Arquivos, é possível
exportar a sessão para:

- EDF/EDF+ — formato clínico padrão;
- FIF — formato científico (MNE-Python);
- BIDS — estrutura padronizada para compartilhamento de dados;
- PDF — relatório automático (sinal + FFT + bandas por canal),
com veredito de qualidade (APTO / DUVIDOSO / DESCARTAR).

> Dica: 
Toda exportação carimba a proveniência (versão do programa e endereço do
repositório), para que qualquer análise possa ser rastreada até o software que a
gerou.

## Ferramentas

No menu Ferramentas:

- Perfil de protocolo (Exportar/Importar) — salva sua
montagem (mapeamento de eletrodos, tipos de sinal por canal,
ajustes de EMG e limites de impedância) num arquivo .json
reutilizável. Importe-o em outra máquina para reproduzir o mesmo setup
(reabra o programa para aplicar por completo).
- Área Maker — atalho para as receitas de análise.
- Abrir pasta de logs — vai direto para
Documentos/EEG_Coletor/logs (útil para anexar a um relato de
problema).

## Atualizações (opcionais e manuais)

Por padrão o programa funciona 100% offline e não faz nenhuma
conexão automática (privacidade). Quando você quiser, verifique se há uma
versão nova em Ajuda → Verificar atualizações:

- Havendo versão nova, o programa mostra o resumo das mudanças e pergunta se
deseja baixar (. Apenas o código é
trocado (rápido); reabra o programa para usar a nova versão.
- Se estiver offline, aparece uma mensagem amigável e nada é alterado.

de atualização (opcional).

> Dica: 
A verificação só baixa código — nunca envia seus dados. O
download é conferido por assinatura SHA-256 (rejeita arquivo alterado).

## Assistente de ajuda (offline)

Em dúvida? Pressione F1 ou vá em Ajuda → Assistente
de ajuda para abrir um chat que responde sobre o uso do programa
(. Ele busca localmente numa base
de perguntas frequentes e no catálogo de erros — nada sai da sua máquina
(sem IA na nuvem, sem internet). Digite uma pergunta (ex.: como faço
impedância?) ou um código de erro (ex.: E115); há também botões de tema
(Conexão, Calibração, Exportar, Estatística, Erros, Privacidade).
: chat de
perguntas frequentes e códigos de erro, sem enviar dados para fora.

## Solução de problemas

- O Windows bloqueou o programa (SmartScreen).: Clique em Mais
informações → Executar assim mesmo.
- Não aparece nenhuma porta COM.: Verifique o cabo/dongle da Cyton e clique
em Atualizar; ou use o Modo Simulação para testar.
- Sinal ruidoso ou amplitude estranha.: Refaça a calibração de impedância,
melhore o contato dos eletrodos e ative o notch de 60,Hz.
- Onde ficam minhas gravações?: Na subpasta sessions/ ao lado do
executável, uma pasta por sessão.
- O programa apresentou um erro.: Ele mostra um aviso e salva um
relatório em Documentos/EEG_Coletor/logs/app.log. Anexe esse
arquivo ao relatar o problema — ele ajuda a diagnosticar.

## Códigos de erro (Erro E0XX
)
Quando algo dá errado, o programa mostra uma notificação com um código
(ex.: Erro E001) e uma explicação do que aconteceu e do que fazer
(. Os códigos seguem faixas:
E0xx conexão, E1xx arquivos/exportação, E2xx análise,
E3xx configuração, E4xx atualização, E5xx dependências,
E6xx interface e E999 (erro inesperado). A lista completa está
no arquivo ERROS.md.
): o que aconteceu, o que fazer e o detalhe técnico.

Para conhecer ou testar as mensagens, use Ajuda →
Diagnóstico de erros (simular) (:
selecione um código e clique em Simular notificação.

lime
OpenBiônica — Edição Clínica | OpenBionica
"""


def _help_load_text(fname, embed=""):
    """Le um .md de conhecimento do disco (SCRIPT_DIR/DOC_DIR/_MEIPASS)
    ou, se ausente/vazio, usa a copia EMBUTIDA no proprio .py. Isso torna
    o codigo autossuficiente: a atualizacao a quente (que so troca o .py)
    nao depende de arquivos-satelite."""
    cands = [os.path.join(SCRIPT_DIR, fname), os.path.join(DOC_DIR, fname)]
    if _MEIPASS_DIR:
        cands.append(os.path.join(_MEIPASS_DIR, fname))
    for p in cands:
        try:
            if os.path.exists(p):
                t = open(p, encoding="utf-8").read()
                if t and t.strip():
                    return t
        except Exception:
            pass
    return embed or ""


def _help_read_kb():
    """Le BASE_CONHECIMENTO.md (disco ou EMBUTIDO) e fatia por seções '## '."""
    txt = _help_load_text("BASE_CONHECIMENTO.md", BASE_CONHECIMENTO_EMBED)
    out = []
    for chunk in re.split(r"\n##\s+", txt or ""):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split("\n", 1)
        title = parts[0].lstrip("# ").strip()
        body = parts[1].strip() if len(parts) > 1 else ""
        if len(body) > 20:
            out.append((title, body))
    return out


_HELP_CORPUS = None   # cache: lista de docs {title, answer, ref, kind, tok}
_HELP_IDF = None      # cache: term -> idf


def _help_corpus():
    """Monta (1x) o corpus indexado: FAQ + base de conhecimento (manual) +
    catalogo de erros, com IDF para ranqueamento TF-IDF. Tudo OFFLINE."""
    global _HELP_CORPUS, _HELP_IDF
    if _HELP_CORPUS is not None:
        return _HELP_CORPUS, _HELP_IDF
    docs = []
    for e in HELP_FAQ:                                   # FAQ curada (precisao)
        docs.append({"title": e["t"], "kind": "faq", "ref": e.get("ref", ""),
                     "answer": e["a"],
                     "text": e["k"] + " " + e["t"] + " " + _help_strip_html(e["a"])})
    for title, body in _help_read_kb():                  # manual (cobertura)
        docs.append({"title": title, "kind": "kb", "ref": "Manual do usuário",
                     "answer": _help_kb_to_html(body), "text": title + " " + body})
    for code, (t, msg, _b) in ERROR_CATALOG.items():     # catalogo de erros
        docs.append({"title": f"Erro {code} — {t}", "kind": "err",
                     "ref": "Ajuda → Diagnóstico de erros", "answer": msg,
                     "text": f"{code} {t} {msg}"})
    df = {}
    for d in docs:
        d["tok"] = _help_tokens(d["text"])
        for term in d["tok"]:
            df[term] = df.get(term, 0) + 1
    n = max(1, len(docs))
    _HELP_IDF = {term: math.log((n + 1) / (c + 1)) + 1.0 for term, c in df.items()}
    _HELP_CORPUS = docs
    return _HELP_CORPUS, _HELP_IDF


def help_answer(query, limit=3):
    """Busca OFFLINE por recuperacao TF-IDF sobre FAQ + manual + erros. Retorna
    [(titulo, resposta_html, ref)]. NADA sai da maquina."""
    q = (query or "").strip()
    if not q:
        return []
    m = re.search(r"\bE\d{3}\b", q.upper())              # codigo de erro explicito
    if m and m.group() in ERROR_CATALOG:
        code = m.group(); title, msg, _b = error_info(code)
        return [(f"Erro {code} — {title}", msg, "Ajuda → Diagnóstico de erros")]
    qtok = _help_tokens(q)
    if not qtok:
        return []
    docs, idf = _help_corpus()
    qn = _help_norm(q)
    scored = []
    for d in docs:
        inter = qtok & d["tok"]
        if not inter:
            continue
        score = sum(idf.get(term, 1.0) for term in inter)
        if d["kind"] == "faq":                           # FAQ tem prioridade
            score *= 1.35
            score += difflib.SequenceMatcher(None, qn, _help_norm(d["text"][:120])).ratio()
        if qtok & _help_tokens(d["title"]):              # bonus de titulo
            score += 1.0
        scored.append((score, d))
    if not scored:
        return []
    scored.sort(key=lambda x: x[0], reverse=True)
    if scored[0][0] < 1.2:                               # limiar de relevancia
        return []
    out, seen = [], set()
    for _s, d in scored:
        if d["title"] in seen:
            continue
        seen.add(d["title"])
        out.append((d["title"], d["answer"], d["ref"]))
        if len(out) >= limit:
            break
    return out


# ============================================================
# Consultor metodológico — raciocínio focado sobre GUIA_METODOLOGICO.md
# (offline: nada sai da máquina; respostas embasadas em literatura)
# ============================================================
_GUIA_CACHE = None


def _html_escape(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("'", "&#39;").replace('"', "&quot;"))


def _help_read_guia():
    """Lê GUIA_METODOLOGICO.md e devolve [(titulo, corpo, refs)] por seção '## '.
    Separa a linha '> Refs:' do corpo. Cacheado. Tudo OFFLINE."""
    global _GUIA_CACHE
    if _GUIA_CACHE is not None:
        return _GUIA_CACHE
    txt = _help_load_text("GUIA_METODOLOGICO.md", GUIA_METODOLOGICO_EMBED)
    out = []
    for chunk in re.split(r"\n##\s+", txt or ""):
        chunk = chunk.strip()
        if not chunk or chunk.startswith("#"):
            continue
        parts = chunk.split("\n", 1)
        title = parts[0].lstrip("# ").strip()
        if title.lower().startswith("refer"):
            continue
        body = parts[1].strip() if len(parts) > 1 else ""
        refs = ""
        mref = re.search(r"(?ms)^>\s*Refs?:\s*(.+?)\s*$", body)
        if mref:
            refs = re.sub(r"\s+", " ", mref.group(1)).strip().rstrip(".")
            body = body[:mref.start()].strip()
        if len(body) > 20:
            out.append((title, body, refs))
    _GUIA_CACHE = out
    return out


def _md_to_chat_html(text):
    """Converte um trecho markdown (do guia) em HTML de balão de chat:
    **negrito**, *itálico*, listas numeradas/marcadas e citações [Autor ANO]."""
    import html as _html
    acc = COLORS.get("accent", "#0f9d75")

    def fmt(t):
        t = _html.escape(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)
        t = re.sub(r"\[([^\]\[]*?\d{4}[^\]\[]*?)\]",
                   r"<span style='color:%s'>[\1]</span>" % acc, t)
        return t

    html, in_ul = [], False
    for ln in text.split("\n"):
        s = ln.rstrip()
        if not s.strip() or s.lstrip().startswith(">"):
            continue
        m_num = re.match(r"^\s*(\d+)\.\s+(.*)", s)
        m_bul = re.match(r"^\s*[-•]\s+(.*)", s)
        if m_num:
            if in_ul:
                html.append("</ul>"); in_ul = False
            html.append("<div style='margin:3px 0'><b>%s.</b> %s</div>"
                        % (m_num.group(1), fmt(m_num.group(2))))
        elif m_bul:
            if not in_ul:
                html.append("<ul style='margin:3px 0 3px 4px'>"); in_ul = True
            html.append("<li style='margin:2px 0'>%s</li>" % fmt(m_bul.group(1)))
        else:
            if in_ul:
                html.append("</ul>"); in_ul = False
            html.append("<div style='margin:4px 0'>%s</div>" % fmt(s))
    if in_ul:
        html.append("</ul>")
    return "".join(html)


# Metadados por seção do guia: "lead" (frase de abertura, dá foco) e
# perguntas de aprofundamento clicáveis. 'match' liga ao título da seção.
_GUIA_META = [
    {"match": "imaginação motora de membro superior",
     "hint": "abordar abordagem estudo desenho protocolo planejar comecar iniciar "
             "membro superior imagetica imaginacao motora reabilitacao roteiro "
             "montagem eletrodo pipeline como comeco por onde",
     "strong": "abordar abordagem metodologia protocolo estudo membro imagetica",
     "lead": "Entendi — você quer <b>planejar um estudo de imaginação motora de "
             "membro superior</b>. Vou por partes, do desenho ao resultado:",
     "follow": [("Detalhar ERD/ERS", "o que é ERD/ERS e como calcular"),
                ("Quantos trials?", "cinestésica vs visual e quantos trials"),
                ("Limpar artefatos", "como limpar artefatos piscadas ICA"),
                ("Validar sem viés", "classificação e validação sem vazamento")]},
    {"match": "ERD/ERS",
     "hint": "erd ers dessincronizacao ressincronizacao rebound beta rebound "
             "band power baseline referencia lateralizacao c3 c4 calcular potencia mu",
     "strong": "erd ers dessincronizacao rebound",
     "lead": "ERD/ERS é o <b>marcador fisiológico central</b> da imaginação "
             "motora. O que é e como calcular:",
     "follow": [("Fazer no programa", "onde calculo ERS/ERD no programa"),
                ("Qual baseline usar", "qual janela de baseline referência usar"),
                ("Roteiro completo", "como abordar estudo de membro superior")]},
    {"match": "Cinestésica vs visual",
     "hint": "cinestesica visual trials tentativas quantos numero classe fadiga "
             "instruir sentir movimento 72 40 imagetica modo primeira pessoa",
     "strong": "cinestesica trials tentativas quantos",
     "lead": "Duas decisões de desenho: <b>tipo de imaginação</b> e <b>nº de "
             "trials</b>. A recomendação:",
     "follow": [("Como instruir", "como instruir o voluntário na cinestésica"),
                ("Roteiro completo", "como abordar estudo de membro superior"),
                ("ERD/ERS", "o que é ERD/ERS e como calcular")]},
    {"match": "Artefatos",
     "hint": "artefato artefatos piscada piscadas olho ocular eog musculo emg "
             "rede notch ruido limpar ica remover contaminacao frontal fp1 fp2",
     "strong": "artefato piscada ocular ica musculo notch",
     "lead": "Para <b>limpar artefatos</b> (piscadas, músculo, rede) sem "
             "distorcer o sinal:",
     "follow": [("Rodar ICA no programa", "como rodar o ICA no programa"),
                ("O que é ICA", "o que é ICA análise de componentes independentes"),
                ("Marcar trials ruins", "como marcar e excluir trials ruins")]},
    {"match": "Classificação e validação",
     "hint": "classificar classificacao validar validacao vazamento leakage csp "
             "fbcsp lda groupkfold loso cross session acuracia kappa treino teste "
             "generalizacao inflar sujeitos",
     "strong": "vazamento leakage csp loso groupkfold classificacao validacao",
     "lead": "A parte que mais <b>infla resultado</b> quando feita errado. "
             "Linha de base e como validar direito:",
     "follow": [("Treinar BCI no programa", "como treino o classificador BCI"),
                ("O que é vazamento", "o que é vazamento de dados leakage"),
                ("Coletar em ≥2 dias", "por que coletar em mais de um dia")]},
    {"match": "topografia, LORETA",
     "hint": "topografia topomap loreta sloreta fonte localizacao laplaciano csd "
             "current source density conducao volume mne eeglab comparar visual "
             "mapa cabeca scalp tomografia",
     "strong": "loreta sloreta topografia csd laplaciano fonte",
     "lead": "Sobre <b>análise visual</b> e a comparação honesta com LORETA / "
             "EEGLAB / MNE:",
     "follow": [("Topografia no programa", "onde vejo a topografia no programa"),
                ("O que é CSD/Laplaciano", "o que é laplaciano de superfície CSD"),
                ("Exportar p/ MNE", "como exporto para MNE ou EEGLAB")]},
]


def consultor_answer(query):
    """Consultor metodológico: classifica a INTENÇÃO e devolve UMA resposta
    focada (não despeja trechos). Retorna dict ou None. Base: GUIA + literatura."""
    secs = _help_read_guia()
    if not secs:
        return None
    qtok = _help_tokens(query)
    if not qtok:
        return None
    best, best_score = None, 0
    for meta in _GUIA_META:
        score = len(qtok & _help_tokens(meta["hint"]))
        score += 2 * len(qtok & _help_tokens(meta.get("strong", "")))
        if score > best_score:
            best_score, best = score, meta
    if not best or best_score < 1:
        return None
    sec = next((s for s in secs
                if _help_norm(best["match"]) in _help_norm(s[0])), None)
    if not sec:
        return None
    title, body, refs = sec
    return {"kind": "guia", "title": title, "lead": best["lead"],
            "body_html": _md_to_chat_html(body), "refs": refs,
            "follow": best["follow"], "score": best_score}


class HelpAssistantDialog(QtWidgets.QDialog):
    """Consultor OpenBionica (OFFLINE): raciocina sobre a intenção e responde de
    forma FOCADA — metodologia embasada (GUIA + literatura), uso do software
    (FAQ/manual) e erros (catálogo). NENHUMA informação sai da máquina."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Consultor OpenBionica (offline)")
        self.setMinimumSize(720, 640)
        v = QtWidgets.QVBoxLayout(self)
        self.view = QtWidgets.QTextBrowser()
        self.view.setOpenLinks(False)          # tratamos os links (ask:/http)
        self.view.setOpenExternalLinks(False)
        self.view.anchorClicked.connect(self._on_anchor)
        v.addWidget(self.view, 1)
        # atalhos por tema (metodologia + software)
        chips = QtWidgets.QHBoxLayout()
        for label, q in (("Planejar estudo (MS)", "como abordar estudo de imaginação "
                          "motora de membro superior"),
                         ("ERD/ERS", "o que é ERD/ERS e como calcular"),
                         ("Artefatos/ICA", "como limpar artefatos piscadas ICA"),
                         ("Validação", "classificação e validação sem vazamento"),
                         ("Topografia/LORETA", "topografia LORETA localização de fonte"),
                         ("Erros (E0XX)", "código de erro")):
            b = QtWidgets.QPushButton(label)
            b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, t=q: self._ask(t))
            chips.addWidget(b)
        chips.addStretch(1)
        v.addLayout(chips)
        h = QtWidgets.QHBoxLayout()
        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText(
            "Pergunte como um consultor (ex.: 'como abordo EEG de membro superior?' · "
            "'qual banda para MI?' · 'E115')…")
        self.input.returnPressed.connect(self._on_send)
        btn = QtWidgets.QPushButton("Perguntar"); btn.clicked.connect(self._on_send)
        btn.setDefault(True)
        h.addWidget(self.input, 1); h.addWidget(btn)
        v.addLayout(h)
        try: self.setStyleSheet(build_stylesheet(COLORS))
        except Exception: pass
        self._html = []
        self._intro()

    # ---------- apresentação (balões) ----------
    def _intro(self):
        self._say("assistant",
                  "Olá! Sou o <b>Consultor OpenBionica</b> — trabalho <b>100% offline</b> "
                  "(nada sai da sua máquina). Diferente de uma busca, eu tento <b>entender "
                  "sua intenção</b> e dar uma resposta <b>focada e embasada</b> (com "
                  "referências).<br><br>Posso ajudar com <b>metodologia</b> (desenho de "
                  "estudo, ERD/ERS, artefatos, validação, análise visual), <b>uso do "
                  "programa</b> e <b>erros</b> (digite o código, ex.: <b>E115</b>).<br><br>"
                  "Por onde começamos?",
                  follow=[("Planejar um estudo de MS",
                           "como abordar estudo de imaginação motora de membro superior"),
                          ("Qual banda/janela para MI",
                           "qual banda e janela usar para imaginação motora"),
                          ("Comparar antes × depois",
                           "como comparar antes e depois estatística")])

    def _say(self, who, html, follow=None, refs=None):
        acc = COLORS.get("accent", "#0f9d75")
        surf = COLORS.get("surface_alt", "#efefef")
        usr = COLORS.get("surface", "#ffffff")
        dim = COLORS.get("text_dim", "#777")
        inner = html
        if refs:
            inner += (f"<div style='margin-top:8px;color:{dim};font-size:11px'>"
                      f"📚 <i>{refs}</i></div>")
        if follow:
            links = " &nbsp;·&nbsp; ".join(
                f"<a href='ask:{_html_escape(q)}' style='color:{acc};"
                f"text-decoration:none'>▸ {_html_escape(lbl)}</a>"
                for lbl, q in follow)
            inner += (f"<div style='margin-top:9px'>"
                      f"<span style='color:{dim};font-size:11px'>Aprofundar:</span><br>"
                      f"{links}</div>")
        if who == "assistant":
            bubble = (
                f"<table cellspacing='0' cellpadding='0' width='88%'><tr>"
                f"<td width='4' bgcolor='{acc}'></td>"
                f"<td bgcolor='{surf}' style='padding:9px 12px'>"
                f"<b style='color:{acc}'>🤝 Consultor</b><br>{inner}</td></tr></table>")
            align = "left"
        else:
            bubble = (
                f"<table cellspacing='0' cellpadding='0' width='78%'><tr>"
                f"<td bgcolor='{usr}' style='padding:8px 12px;border:1px solid {surf}'>"
                f"<b style='color:{dim}'>Você</b><br>{inner}</td></tr></table>")
            align = "right"
        self._html.append(
            f"<table width='100%' cellspacing='0' cellpadding='0' "
            f"style='margin:7px 0'><tr><td align='{align}'>{bubble}</td></tr></table>")
        self.view.setHtml("".join(self._html))
        sb = self.view.verticalScrollBar(); sb.setValue(sb.maximum())

    def _on_send(self):
        t = self.input.text().strip()
        if t:
            self.input.clear(); self._ask(t)

    def _on_anchor(self, url):
        s = url.toString()
        if s.startswith("ask:"):
            from urllib.parse import unquote
            self._ask(unquote(s[4:]))
        else:
            try: QtGui.QDesktopServices.openUrl(url)
            except Exception: pass

    # ---------- raciocínio: erro → consultor → software → escalonar ----------
    def _ask(self, text):
        self._say("user", text)
        up = text.upper()
        mcode = re.search(r"\bE\d{3}\b", up)
        if mcode and mcode.group() in ERROR_CATALOG:
            code = mcode.group(); title, msg, _b = error_info(code)
            self._say("assistant",
                      f"<b>Erro {code} — {title}</b><br>{msg}", refs=None,
                      follow=[("Ver todos os erros", "código de erro"),
                              ("Onde fica o log", "onde fica o log de erros")])
            return
        # 1) Consultor metodológico (resposta focada e embasada)
        c = consultor_answer(text)
        if c and c["score"] >= 2:
            self._say("assistant",
                      f"{c['lead']}<br><br>{c['body_html']}",
                      refs=c["refs"], follow=c["follow"])
            return
        # 2) Uso do software (FAQ + manual + erros), foco no melhor resultado
        hits = help_answer(text, limit=3)
        if hits:
            title, ans, ref = hits[0]
            dim = COLORS.get("text_dim", "#777")
            body = f"<b>{title}</b><br>{ans}"
            if ref:
                body += f"<br><span style='color:{dim};font-size:11px'>↳ {ref}</span>"
            extra = [(t2, f"{t2}") for t2, _a, _r in hits[1:3]]
            fol = extra + [("Planejar um estudo",
                            "como abordar estudo de membro superior")]
            self._say("assistant", body, follow=fol)
            return
        # 3) fraco no consultor mas relevante? entrega mesmo assim
        if c:
            self._say("assistant",
                      f"{c['lead']}<br><br>{c['body_html']}",
                      refs=c["refs"], follow=c["follow"])
            return
        # 4) Não mapeado → escalonar com honestidade
        acc = COLORS.get("accent", "#0f9d75")
        self._say("assistant",
                  "Ainda <b>não tenho esse tema mapeado</b> na minha base. Para não te "
                  "dar informação genérica, prefiro ser honesto:<br>"
                  "• Reformule com uma palavra-chave (ex.: <i>banda, trials, artefato, "
                  "validação, topografia, exportar</i>), ou use os atalhos acima;<br>"
                  "• Se for específico do seu <b>hardware/método</b>, fale com o "
                  "<b>suporte/orientador</b> da equipe;<br>"
                  f"• Como o OpenBionica é <b style='color:{acc}'>código aberto</b>, você "
                  "(ou sua equipe) pode <b>implementar/corrigir</b> direto no código.",
                  follow=[("Planejar um estudo (MS)",
                           "como abordar estudo de membro superior"),
                          ("Ver temas de metodologia", "metodologia"),
                          ("Diagnóstico de erros", "código de erro")])


# ============================================================
# Area Maker — Receitas de analise (pipeline composavel, salvavel em .json)
# ============================================================
RECIPE_VERSION = "1.0"
RECIPE_BANDS = dict(EEG_BANDS)
RECIPE_BANDS["Mu (8-13)"] = (8.0, 13.0)
RECIPE_METRICS = [("band", "Potência de banda"),
                  ("erd", "ERD% vs baseline (imagética motora)"),
                  ("relband", "Potência relativa da banda (%)"),
                  ("rms", "RMS"),
                  ("ptp", "Amplitude pico-a-pico"), ("std", "Desvio padrão")]

# Métricas que usam o seletor de BANDA
RECIPE_BAND_METRICS = {"band", "erd", "relband"}

# Modelos prontos (estilo "cenários" do OpenVIBE) — embasados no guia.
RECIPE_TEMPLATES = [
    {"nome": "ERD Mu — imagética motora", "metrica": "erd", "banda": "Mu (8-13)",
     "filtro": {"low": 8, "high": 30}, "baseline": {"b0": 0.0, "b1": 2.0},
     "t0": 2.5, "t1": 4.5,
     "_dica": "ERD% na banda mu (8–13 Hz): baseline 0–2 s (repouso) × janela de MI "
              "2,5–4,5 s. Espere ERD negativo em C3/C4 [Pfurtscheller & Lopes da Silva 1999]."},
    {"nome": "ERD Beta — imagética motora", "metrica": "erd", "banda": "Beta",
     "filtro": {"low": 8, "high": 30}, "baseline": {"b0": 0.0, "b1": 2.0},
     "t0": 2.5, "t1": 4.5,
     "_dica": "ERD% em beta (13–30 Hz). Após o fim da MI costuma haver ERS "
              "(beta rebound) — teste também uma janela pós-movimento."},
    {"nome": "Potência relativa Alpha (relaxamento)", "metrica": "relband",
     "banda": "Alpha", "filtro": {"low": 1, "high": 40},
     "_dica": "Alpha relativo (% da potência total 1–40 Hz). Sobe no relaxamento / "
              "olhos fechados, forte em região occipital."},
    {"nome": "Potência relativa Beta (atenção/foco)", "metrica": "relband",
     "banda": "Beta", "filtro": {"low": 1, "high": 40},
     "_dica": "Beta relativo (%): tende a subir com atenção/engajamento."},
    {"nome": "RMS bruto (qualidade do sinal)", "metrica": "rms",
     "_dica": "Amplitude RMS por canal (µV): útil para triagem de canais ruins / "
             "eletrodo solto (valores muito altos ou ~0)."},
]


def run_recipe(recipe, session):
    """Pipeline de analise sobre uma sessao: RECORTE temporal -> PRE-FILTRO
    (passa-banda) -> METRICA por canal. Retorna [(rotulo_canal, valor)]."""
    eeg = np.asarray(session.get("eeg")) if session else None
    if eeg is None or eeg.ndim != 2 or eeg.size == 0:
        return []
    sr = float(session.get("sr", SAMPLE_RATE))
    names = session.get("ch_names") or [f"CH{i+1}" for i in range(eeg.shape[0])]
    metric = recipe.get("metrica", "band")
    band = recipe.get("banda", "Alpha")
    sel = recipe.get("canais")
    idxs = (list(range(eeg.shape[0])) if (not sel or sel == "all")
            else [c for c in sel if 0 <= c < eeg.shape[0]])
    # --- etapa: recorte temporal (segundos) ---
    n_total = eeg.shape[1]
    t0, t1 = recipe.get("t0"), recipe.get("t1")
    i0 = int(max(0.0, float(t0)) * sr) if t0 not in (None, "") else 0
    i1 = int(float(t1) * sr) if t1 not in (None, "") else n_total
    i0 = max(0, min(i0, n_total)); i1 = max(i0, min(i1, n_total))
    # --- etapa: janela de baseline (para ERD%) ---
    bl = recipe.get("baseline") or {}
    b0, b1 = bl.get("b0"), bl.get("b1")
    j0 = int(float(b0) * sr) if b0 not in (None, "") else 0
    j1 = int(float(b1) * sr) if b1 not in (None, "") else int(min(2.0 * sr, n_total))
    j0 = max(0, min(j0, n_total)); j1 = max(j0 + 1, min(j1, n_total))
    hi_tot = min(40.0, 0.49 * sr)   # banda "total" para potência relativa
    # --- etapa: pre-filtro (passa-banda) ---
    sos = None
    filt = recipe.get("filtro")
    if filt:
        try:
            ny = 0.5 * sr
            lo_n = max(1e-4, float(filt.get("low", 0)) / ny)
            hi_n = min(0.999, float(filt.get("high", 0)) / ny)
            if 0 < lo_n < hi_n < 1:
                sos = scipy_signal.butter(4, [lo_n, hi_n], btype="band", output="sos")
        except Exception:
            sos = None
    out = []
    for c in idxs:
        chan = np.asarray(eeg[c], dtype=float)
        sig = chan[i0:i1]; sig = sig[np.isfinite(sig)]
        if sig.size < 2:
            out.append((names[c], 0.0)); continue
        if sos is not None and sig.size > 18:
            try: sig = scipy_signal.sosfiltfilt(sos, sig)
            except Exception: pass
        # --- etapa: metrica ---
        if metric == "band":
            lo, hi = RECIPE_BANDS.get(band, (8.0, 13.0))
            val = SignalProcessor.compute_band_power(sig, lo, hi, sr)
        elif metric == "relband":                        # potência relativa (%)
            lo, hi = RECIPE_BANDS.get(band, (8.0, 13.0))
            pb = SignalProcessor.compute_band_power(sig, lo, hi, sr)
            pt = SignalProcessor.compute_band_power(sig, 1.0, hi_tot, sr)
            val = 100.0 * pb / pt if pt > 0 else 0.0
        elif metric == "erd":                            # ERD% vs baseline
            lo, hi = RECIPE_BANDS.get(band, (8.0, 13.0))
            base = chan[j0:j1]; base = base[np.isfinite(base)]
            if sos is not None and base.size > 18:
                try: base = scipy_signal.sosfiltfilt(sos, base)
                except Exception: pass
            p_act = SignalProcessor.compute_band_power(sig, lo, hi, sr)
            p_base = (SignalProcessor.compute_band_power(base, lo, hi, sr)
                      if base.size >= 2 else 0.0)
            val = 100.0 * (p_act - p_base) / p_base if p_base > 0 else 0.0
        elif metric == "rms":
            val = float(np.sqrt(np.mean(sig ** 2)))
        elif metric == "ptp":
            val = float(np.ptp(sig))
        else:
            val = float(np.std(sig))
        out.append((names[c], float(val)))
    return out


class RecipeDialog(QtWidgets.QDialog):
    """Area Maker (v1): monta uma RECEITA de analise (metrica + banda + canais),
    roda em uma ou mais sessoes e salva/carrega como .json reutilizavel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._win = parent
        self._sessions = []   # [(nome, dict)]
        self.setWindowTitle("Receitas de análise — Área Maker")
        self.setMinimumSize(780, 600)
        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(QtWidgets.QLabel(
            "<b>Monte uma receita de análise</b> (recorte → filtro → métrica) e rode "
            "em uma ou várias sessões. Métricas incluem <b>ERD%</b> e <b>potência "
            "relativa</b>. Salve como <b>.json</b> para reutilizar/compartilhar o "
            "protocolo (reprodutibilidade). Tudo offline."))
        form = QtWidgets.QFormLayout()
        # ---- modelos prontos (estilo "cenários" do OpenVIBE) ----
        tpl_row = QtWidgets.QHBoxLayout()
        self.tpl_combo = QtWidgets.QComboBox()
        self.tpl_combo.addItem("— escolher um modelo —", None)
        for t in RECIPE_TEMPLATES:
            self.tpl_combo.addItem(t["nome"], t)
        tpl_btn = QtWidgets.QPushButton("Aplicar modelo")
        tpl_btn.clicked.connect(self._apply_template)
        tpl_row.addWidget(self.tpl_combo, 1); tpl_row.addWidget(tpl_btn)
        form.addRow("Modelo pronto:", tpl_row)
        self.tpl_hint = QtWidgets.QLabel("")
        self.tpl_hint.setWordWrap(True)
        self.tpl_hint.setStyleSheet(f"color:{COLORS.get('text_dim','#777')};font-size:11px")
        form.addRow("", self.tpl_hint)
        self.name_edit = QtWidgets.QLineEdit("Minha análise")
        form.addRow("Nome da receita:", self.name_edit)
        self.metric_combo = QtWidgets.QComboBox()
        for code, label in RECIPE_METRICS:
            self.metric_combo.addItem(label, code)
        self.metric_combo.currentIndexChanged.connect(self._sync_band)
        form.addRow("Métrica:", self.metric_combo)
        self.band_combo = QtWidgets.QComboBox()
        for b in RECIPE_BANDS:
            self.band_combo.addItem(b, b)
        self.band_combo.setCurrentText("Alpha")
        form.addRow("Banda:", self.band_combo)
        self.chan_edit = QtWidgets.QLineEdit()
        self.chan_edit.setPlaceholderText("vazio = todos os canais  ·  ou ex.: 1,2,3 (1-based)")
        form.addRow("Canais:", self.chan_edit)
        # etapa do pipeline: recorte temporal
        cut = QtWidgets.QHBoxLayout()
        self.t0_edit = QtWidgets.QLineEdit(); self.t0_edit.setPlaceholderText("início (s)")
        self.t1_edit = QtWidgets.QLineEdit(); self.t1_edit.setPlaceholderText("fim (s)")
        cut.addWidget(self.t0_edit); cut.addWidget(QtWidgets.QLabel("até"))
        cut.addWidget(self.t1_edit)
        form.addRow("Recorte (vazio = tudo):", cut)
        # etapa do pipeline: pre-filtro (passa-banda)
        fl = QtWidgets.QHBoxLayout()
        self.filt_chk = QtWidgets.QCheckBox("Pré-filtrar")
        self.filt_lo = QtWidgets.QLineEdit("1");  self.filt_lo.setMaximumWidth(60)
        self.filt_hi = QtWidgets.QLineEdit("40"); self.filt_hi.setMaximumWidth(60)
        fl.addWidget(self.filt_chk); fl.addWidget(self.filt_lo)
        fl.addWidget(QtWidgets.QLabel("–")); fl.addWidget(self.filt_hi)
        fl.addWidget(QtWidgets.QLabel("Hz")); fl.addStretch(1)
        form.addRow("Pré-filtro (passa-banda):", fl)
        # etapa do pipeline: baseline (só para ERD%)
        bl = QtWidgets.QHBoxLayout()
        self.b0_edit = QtWidgets.QLineEdit("0");   self.b0_edit.setMaximumWidth(60)
        self.b1_edit = QtWidgets.QLineEdit("2");   self.b1_edit.setMaximumWidth(60)
        bl.addWidget(self.b0_edit); bl.addWidget(QtWidgets.QLabel("até"))
        bl.addWidget(self.b1_edit); bl.addWidget(QtWidgets.QLabel("s (repouso)"))
        bl.addStretch(1)
        self.baseline_row = QtWidgets.QWidget(); self.baseline_row.setLayout(bl)
        form.addRow("Baseline p/ ERD%:", self.baseline_row)
        v.addLayout(form)
        sh = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Adicionar sessões…")
        add_btn.clicked.connect(self._add_sessions)
        clr_btn = QtWidgets.QPushButton("Limpar")
        clr_btn.clicked.connect(self._clear_sessions)
        self.sess_lbl = QtWidgets.QLabel("0 sessões")
        sh.addWidget(add_btn); sh.addWidget(clr_btn); sh.addWidget(self.sess_lbl)
        sh.addStretch(1)
        v.addLayout(sh)
        self.result_lbl = QtWidgets.QLabel("")
        self.result_lbl.setStyleSheet(f"color:{COLORS.get('accent','#0f9d75')};"
                                      "font-weight:bold")
        v.addWidget(self.result_lbl)
        self.table = QtWidgets.QTableWidget(0, 1)
        self.table.setHorizontalHeaderLabels(["Canal"])
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        v.addWidget(self.table, 1)
        ah = QtWidgets.QHBoxLayout()
        run_btn = QtWidgets.QPushButton("▶ Rodar"); run_btn.clicked.connect(self._run)
        save_btn = QtWidgets.QPushButton("Salvar receita…"); save_btn.clicked.connect(self._save_recipe)
        load_btn = QtWidgets.QPushButton("Carregar receita…"); load_btn.clicked.connect(self._load_recipe)
        exp_btn = QtWidgets.QPushButton("Exportar tabela…"); exp_btn.clicked.connect(self._export_csv)
        close_btn = QtWidgets.QPushButton("Fechar"); close_btn.clicked.connect(self.accept)
        for b in (run_btn, save_btn, load_btn, exp_btn):
            ah.addWidget(b)
        ah.addStretch(1); ah.addWidget(close_btn)
        v.addLayout(ah)
        try: self.setStyleSheet(build_stylesheet(COLORS))
        except Exception: pass
        self._sync_band()

    def _sync_band(self):
        metric = self.metric_combo.currentData()
        self.band_combo.setEnabled(metric in RECIPE_BAND_METRICS)
        self.baseline_row.setEnabled(metric == "erd")

    def _apply_template(self):
        """Preenche o formulário a partir de um modelo pronto (RECIPE_TEMPLATES)."""
        t = self.tpl_combo.currentData()
        if not t:
            self.tpl_hint.setText(""); return
        self.name_edit.setText(t["nome"])
        i = self.metric_combo.findData(t.get("metrica", "band"))
        if i >= 0:
            self.metric_combo.setCurrentIndex(i)
        j = self.band_combo.findData(t.get("banda", "Alpha"))
        if j >= 0:
            self.band_combo.setCurrentIndex(j)
        self.chan_edit.setText("")                     # modelos: todos os canais
        self.t0_edit.setText("" if t.get("t0") in (None, "") else str(t["t0"]))
        self.t1_edit.setText("" if t.get("t1") in (None, "") else str(t["t1"]))
        filt = t.get("filtro") or {}
        self.filt_chk.setChecked(bool(filt))
        if filt:
            self.filt_lo.setText(str(filt.get("low", "1")))
            self.filt_hi.setText(str(filt.get("high", "40")))
        base = t.get("baseline") or {}
        if base:
            self.b0_edit.setText(str(base.get("b0", "0")))
            self.b1_edit.setText(str(base.get("b1", "2")))
        self.tpl_hint.setText("💡 " + t.get("_dica", ""))
        self._sync_band()

    def _recipe(self):
        chans = "all"
        txt = self.chan_edit.text().strip()
        if txt:
            try:
                chans = [int(x) - 1 for x in re.split(r"[,;\s]+", txt) if x.strip()]
            except Exception:
                chans = "all"
        def _num(le):
            t = le.text().strip().replace(",", ".")
            try: return float(t) if t else None
            except Exception: return None
        rec = {"recipe_version": RECIPE_VERSION, "nome": self.name_edit.text().strip(),
               "metrica": self.metric_combo.currentData(),
               "banda": self.band_combo.currentData(), "canais": chans,
               "t0": _num(self.t0_edit), "t1": _num(self.t1_edit)}
        if self.filt_chk.isChecked():
            lo, hi = _num(self.filt_lo), _num(self.filt_hi)
            if lo is not None and hi is not None and hi > lo:
                rec["filtro"] = {"low": lo, "high": hi}
        if self.metric_combo.currentData() == "erd":
            rec["baseline"] = {"b0": _num(self.b0_edit), "b1": _num(self.b1_edit)}
        return rec

    def _add_sessions(self):
        if not hasattr(self._win, "_load_session_csv"):
            return
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Selecionar sessões (CSV)", "", "CSV (*.csv)")
        for p in paths:
            try:
                d = self._win._load_session_csv(p)
                if d:
                    self._sessions.append((os.path.basename(p), d))
            except Exception as exc:
                logging.getLogger("eeg").warning("receita: falha em %s: %s", p, exc)
        self.sess_lbl.setText(f"{len(self._sessions)} sessões")

    def _clear_sessions(self):
        self._sessions = []
        self.sess_lbl.setText("0 sessões")
        self.table.setRowCount(0); self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels(["Canal"])

    def _run(self):
        if not self._sessions:
            QtWidgets.QMessageBox.information(
                self, "Receita", "Adicione ao menos uma sessão antes de rodar.")
            return
        rec = self._recipe()
        results = [(nm, run_recipe(rec, d)) for nm, d in self._sessions]
        if not results[0][1]:
            QtWidgets.QMessageBox.information(
                self, "Receita", "A receita não produziu resultados nessa sessão.")
            return
        metric = rec.get("metrica", "band")
        unit = {"erd": "ERD% (negativo = dessincronização)",
                "relband": "% da potência total (1–40 Hz)",
                "band": "potência de banda (µV²)", "rms": "µV (RMS)",
                "ptp": "µV (pico-a-pico)", "std": "µV (desvio padrão)"}.get(metric, "")
        mlabel = dict(RECIPE_METRICS).get(metric, metric)
        bandtxt = (f" · banda {rec.get('banda')}" if metric in RECIPE_BAND_METRICS
                   else "")
        self.result_lbl.setText(f"Métrica: {mlabel}{bandtxt}  —  unidade: {unit}")
        chan_order = [c for c, _ in results[0][1]]
        self.table.setColumnCount(1 + len(results))
        self.table.setHorizontalHeaderLabels(["Canal"] + [nm for nm, _ in results])
        self.table.setRowCount(len(chan_order))
        for r, ch in enumerate(chan_order):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(ch))
            for ci, (_nm, rows) in enumerate(results):
                val = dict(rows).get(ch)
                it = QtWidgets.QTableWidgetItem(f"{val:.4g}" if isinstance(val, float) else "—")
                it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, 1 + ci, it)
        self.table.resizeColumnsToContents()

    def _save_recipe(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Salvar receita",
            (self.name_edit.text().strip() or "receita") + ".json", "Receita (*.json)")
        if not p:
            return
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self._recipe(), f, ensure_ascii=False, indent=2)
            QtWidgets.QMessageBox.information(self, "Receita", f"Receita salva:\n{p}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Receita", f"Falha ao salvar: {exc}")

    def _load_recipe(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Carregar receita", "", "Receita (*.json)")
        if not p:
            return
        try:
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
            self.name_edit.setText(str(rec.get("nome", "")))
            i = self.metric_combo.findData(rec.get("metrica", "band"))
            if i >= 0:
                self.metric_combo.setCurrentIndex(i)
            j = self.band_combo.findData(rec.get("banda", "Alpha"))
            if j >= 0:
                self.band_combo.setCurrentIndex(j)
            ch = rec.get("canais", "all")
            self.chan_edit.setText("" if ch in (None, "all") else
                                   ",".join(str(int(c) + 1) for c in ch))
            self.t0_edit.setText("" if rec.get("t0") in (None, "") else str(rec["t0"]))
            self.t1_edit.setText("" if rec.get("t1") in (None, "") else str(rec["t1"]))
            filt = rec.get("filtro") or {}
            self.filt_chk.setChecked(bool(filt))
            if filt:
                self.filt_lo.setText(str(filt.get("low", "1")))
                self.filt_hi.setText(str(filt.get("high", "40")))
            base = rec.get("baseline") or {}
            if base:
                self.b0_edit.setText("" if base.get("b0") in (None, "") else str(base["b0"]))
                self.b1_edit.setText("" if base.get("b1") in (None, "") else str(base["b1"]))
            self._sync_band()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Receita", f"Falha ao carregar: {exc}")

    def _export_csv(self):
        if self.table.rowCount() == 0:
            QtWidgets.QMessageBox.information(self, "Receita", "Rode a receita primeiro.")
            return
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exportar tabela", "resultado_receita.csv", "CSV (*.csv)")
        if not p:
            return
        try:
            cols = [self.table.horizontalHeaderItem(c).text()
                    for c in range(self.table.columnCount())]
            with open(p, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f); w.writerow(cols)
                for r in range(self.table.rowCount()):
                    w.writerow([self.table.item(r, c).text() if self.table.item(r, c) else ""
                                for c in range(self.table.columnCount())])
            QtWidgets.QMessageBox.information(self, "Receita", f"Tabela exportada:\n{p}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Receita", f"Falha: {exc}")


# ============================================================
# Logging de aplicacao + recuperacao de crash (qualidade comercial)
# ============================================================
def _setup_logging():
    """Logging de SAUDE/ERROS do software (separado do audit de protocolo).
    RotatingFileHandler em DOC_DIR/logs/app.log (5MB x3) + console."""
    logger = logging.getLogger("eeg")
    if logger.handlers:                       # idempotente
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    try:
        log_dir = os.path.join(DOC_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, "app.log"),
            maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception as exc:
        print(f"[logging] nao foi possivel criar o arquivo de log: {exc}")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def _install_excepthook(logger):
    """Handler GLOBAL de excecoes: em vez de o app cair em silencio (no .exe nao
    ha console), loga o traceback completo e mostra um dialogo amigavel com a
    opcao de ver os detalhes/abrir o relatorio."""
    def _hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.critical("Excecao nao tratada:",
                        exc_info=(exc_type, exc_value, exc_tb))
        try:
            short = "".join(
                traceback.format_exception_only(exc_type, exc_value)).strip()
            # Sinaliza como "Erro E999" (catalogo), com traceback nos detalhes
            notify_error("E999", detail=short, exc=exc_value)
        except Exception:
            pass
    sys.excepthook = _hook
    try:
        import faulthandler
        faulthandler.enable()                 # travamentos em C (numpy/scipy/Qt)
    except Exception:
        pass


# ============================================================
# Entry point
# ============================================================
def main():
    # reaproveita a QApplication se o launcher (auto-update) já tiver criado uma
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    # Fonte global da interface (Inter — UI). Widgets de dados redefinem para JetBrains Mono via QSS.
    app.setFont(QtGui.QFont(FONT_UI, 10))

    # Logging de aplicacao + rede de seguranca contra crash silencioso
    logger = _setup_logging()
    logger.info("Iniciando %s v%s", APP_NAME, APP_VERSION)
    _install_excepthook(logger)

    # ----------------------------------------------------------------
    # IMPORTANTE: carrega config.json e aplica o tema salvo ANTES de
    # qualquer setStyleSheet/setPalette. Sem isso, a primeira pintura
    # da janela usa Lime (default) e so depois "salta" para o tema
    # salvo — causando flash visual na parte superior (header).
    # ----------------------------------------------------------------
    early_config = AppConfig()
    _apply_theme_colors(early_config.theme)
    # Idioma persistido (afeta títulos de abas, botões principais)
    I18N.set_language(getattr(early_config, "language", "pt"))
    # Garante que temas personalizados ja estejam em THEMES nesse momento
    for _name, _palette in early_config.custom_themes.items():
        THEMES[_name] = _palette
    # Se o tema salvo for um custom, aplica de fato suas cores
    if early_config.theme in THEMES:
        _apply_theme_colors(early_config.theme)

    # ---- Assistente de primeiro uso (idioma -> layout -> aceite do Termo) ----
    # Reaparece se o termo nunca foi aceito OU se a versao do termo mudou.
    needs_wizard = ((not early_config.terms_accepted)
                    or (early_config.terms_version != TERMS_VERSION))
    if needs_wizard and "--no-wizard" not in sys.argv:
        app.setStyleSheet(build_stylesheet(COLORS))   # estilo p/ o assistente
        wiz = FirstRunWizard(early_config)
        if wiz.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            logger.info("Termo de uso recusado pelo usuário — encerrando.")
            sys.exit(0)                                # sem consentimento -> nao abre
        # O assistente pode ter trocado idioma/layout e ja salvou o config.
        I18N.set_language(getattr(early_config, "language", "pt"))
        _apply_theme_colors(early_config.theme)

    # Configura pyqtgraph com cores ja corretas (evita flash em plots)
    pg.setConfigOption("background", COLORS["background"])
    pg.setConfigOption("foreground", COLORS["text"])
    pg.setConfigOptions(antialias=True)

    # Agora sim: stylesheet e paleta com o tema salvo
    app.setStyleSheet(build_stylesheet(COLORS))
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window,          QtGui.QColor(COLORS["background"]))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText,      QtGui.QColor(COLORS["text"]))
    palette.setColor(QtGui.QPalette.ColorRole.Base,            QtGui.QColor(COLORS["surface_alt"]))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase,   QtGui.QColor(COLORS["surface"]))
    palette.setColor(QtGui.QPalette.ColorRole.Text,            QtGui.QColor(COLORS["text"]))
    palette.setColor(QtGui.QPalette.ColorRole.Button,          QtGui.QColor(COLORS["surface_alt"]))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText,      QtGui.QColor(COLORS["accent"]))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight,       QtGui.QColor(COLORS["accent_dim"]))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(COLORS["background"]))
    app.setPalette(palette)

    # ----------------------------------------------------------------
    # LAUNCHER SCREEN — Tela inicial / dashboard antes da janela principal
    # ----------------------------------------------------------------
    # Pode ser pulada via flag --no-launcher (útil para depuração / atalho).
    skip_launcher = "--no-launcher" in sys.argv
    launcher_choice = None
    if not skip_launcher:
        # VolunteerRegistry temporário para popular o combo
        try:
            temp_vols = VolunteerRegistry(early_config.save_directory)
        except Exception:
            temp_vols = None
        launcher = LauncherScreen(config=early_config, volunteers_mgr=temp_vols)
        result = launcher.exec()
        if result != QtWidgets.QDialog.DialogCode.Accepted:
            # Usuário fechou ou clicou em Sair
            print("[Launcher] Encerrado pelo usuário (sem escolher fluxo).")
            sys.exit(0)
        launcher_choice = launcher.get_choice()

    # Re-aplica o stylesheet principal (o launcher tem o seu próprio QSS).
    # Idempotente: aqui a janela ainda nao existe, entao e barato; mas mantemos
    # o padrao para nao re-polir desnecessariamente.
    _qss = build_stylesheet(COLORS)
    if app.styleSheet() != _qss:
        app.setStyleSheet(_qss)

    window = EEGCollectorWindow()
    # Aplica a escolha do launcher (porta, modo, expansão, tipo de aquisição)
    if launcher_choice is not None:
        window.apply_launcher_choice(launcher_choice)
    # Garantia extra: re-aplica o stylesheet SO se mudou (caso a janela tenha
    # re-lido o config e trocado o tema). Idempotente evita re-polish de ~16s
    # sobre a janela ja construida — principal causa da lentidao de arranque.
    _qss = build_stylesheet(COLORS)
    if app.styleSheet() != _qss:
        app.setStyleSheet(_qss)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
