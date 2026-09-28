"""Industrial Quality Control Dashboard for Steel Welding Inspection.

Provides a Streamlit interface for plant operators to upload weld seam images,
execute real-time CNN inference, inspect probability distributions, and obtain
automated production dispatch decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Final, List, Tuple

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

MODEL_PATH: Final[Path] = Path("modelo_soldadura.h5")
IMAGE_SIZE: Final[Tuple[int, int]] = (224, 224)

# Classes aligned with directory alphabetization in training set
CLASS_KEYS: Final[List[str]] = [
    "ColaRata",
    "DesfaseDosMediaPulgada",
    "DesfaseMediaPulgada",
    "DesfaseTresPulgadas",
    "DesfaseUnaPulgada",
    "LaminaOk",
    "LaminaPerforada",
]

CLASS_DISPLAY_NAMES: Final[Dict[str, str]] = {
    "ColaRata": "Defecto: Cola de Rata",
    "DesfaseDosMediaPulgada": "Desfase Severo (2.5\")",
    "DesfaseMediaPulgada": "Desfase Leve (0.5\")",
    "DesfaseTresPulgadas": "Desfase Crítico (3.0\")",
    "DesfaseUnaPulgada": "Desfase Moderado (1.0\")",
    "LaminaOk": "Conforme (Lámina OK)",
    "LaminaPerforada": "Defecto Crítico: Perforación",
}

SEVERITY_MAPPING: Final[Dict[str, Tuple[str, str]]] = {
    "LaminaOk": ("APROBADO", "success"),
    "DesfaseMediaPulgada": ("REVISIÓN MENOR", "warning"),
    "DesfaseUnaPulgada": ("RECHAZO - AJUSTE REQUERIDO", "error"),
    "DesfaseDosMediaPulgada": ("RECHAZO CRÍTICO", "error"),
    "DesfaseTresPulgadas": ("RECHAZO CRÍTICO", "error"),
    "ColaRata": ("RECHAZO POR POROSIDAD/DISCONTINUIDAD", "error"),
    "LaminaPerforada": ("RECHAZO CRÍTICO - DAÑO ESTRUCTURAL", "error"),
}


@st.cache_resource(show_spinner="Cargando pesos del modelo convolucional...")
def load_welding_model(path: Path) -> tf.keras.Model | None:
    """Loads and caches the compiled Keras model from disk.

    Args:
        path: Path to serialized model file.

    Returns:
        tf.keras.Model or None if file does not exist.
    """
    if not path.exists():
        return None
    return tf.keras.models.load_model(str(path))


def preprocess_image(image: Image.Image, target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Preprocesses a PIL Image into a normalized tensor ready for model inference.

    Args:
        image: Source PIL Image.
        target_size: Desired resolution (width, height).

    Returns:
        np.ndarray: Batch tensor of shape (1, height, width, 3).
    """
    rgb_image = image.convert("RGB")
    resized_image = rgb_image.resize(target_size)
    img_array = tf.keras.utils.img_to_array(resized_image)
    return np.expand_dims(img_array, axis=0)


def setup_ui() -> None:
    """Configures dashboard metadata, layout, and sidebar."""
    st.set_page_config(
        page_title="Control de Calidad de Soldadura | Steel Plant AI",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("🏭 Centro de Monitoreo")
    st.sidebar.markdown(
        """
        **Sistema de Inspección Automatizada**
        - **Modelo**: CNN Custom 3-Bloques
        - **Pipeline**: Inferencia en tiempo real
        - **Dataset**: Aumento con Ruido Gaussiano
        - **Resolución**: 224x224 px
        """
    )
    st.sidebar.divider()
    st.sidebar.info(
        "Este dashboard clasifica uniones soldadas y detecta discontinuidades "
        "estructurales para la prevención de fallas mecánicas en línea de producción."
    )


def main() -> None:
    """Main rendering loop for the Streamlit dashboard."""
    setup_ui()

    st.title("🛡️ Sistema Inteligente de Control de Calidad de Soldadura")
    st.caption("Clasificación de cordones de soldadura y detección de anomalías mediante Visión por Computadora.")

    model = load_welding_model(MODEL_PATH)
    if model is None:
        st.error(
            f"⚠️ Archivo del modelo `{MODEL_PATH}` no encontrado. "
            "Ejecute `python etapa2_entrenamiento.py` para entrenar el modelo antes de iniciar el dashboard."
        )
        return

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("1. Ingesta de Imagen")
        uploaded_file = st.file_uploader(
            "Cargar captura óptica de la unión de soldadura:",
            type=["jpg", "jpeg", "png"],
            help="Soporta formatos estándar capturados por cámaras de inspección industrial.",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Inspección óptica seleccionada", use_container_width=True)

    with col2:
        st.subheader("2. Diagnóstico y Telemetría")

        if uploaded_file is None:
            st.info("👈 Cargue una imagen en el panel izquierdo para iniciar la inferencia.")
            return

        # Preprocessing & Inference
        input_tensor = preprocess_image(image, IMAGE_SIZE)
        raw_predictions = model.predict(input_tensor, verbose=0)

        # Softmax is already present in the model's output layer
        probabilities: np.ndarray = raw_predictions[0]
        predicted_idx: int = int(np.argmax(probabilities))
        confidence_pct: float = float(probabilities[predicted_idx] * 100.0)

        class_key = CLASS_KEYS[predicted_idx]
        display_name = CLASS_DISPLAY_NAMES.get(class_key, class_key)
        status_label, alert_type = SEVERITY_MAPPING.get(class_key, ("INDEFINIDO", "warning"))

        # KPI Metrics Display
        kpi_col1, kpi_col2 = st.columns(2)
        with kpi_col1:
            st.metric(label="Diagnóstico Principal", value=display_name)
        with kpi_col2:
            st.metric(label="Certeza del Modelo", value=f"{confidence_pct:.2f}%")

        # Actionable Industrial Banner
        if alert_type == "success":
            st.success(f"✅ **{status_label}**: Cordón dentro de tolerancias estándar. Aprobado para siguiente fase.")
        elif alert_type == "warning":
            st.warning(f"⚠️ **{status_label}**: Variación detectada. Proceder con ajuste preventivo de antorcha.")
        else:
            st.error(f"🚨 **{status_label}**: Parámetros fuera de norma. Pieza rechazada para reprocesamiento.")

        # Class Probability Distribution
        st.write("---")
        st.markdown("**Distribución de Probabilidad por Clase:**")
        chart_data = {
            CLASS_DISPLAY_NAMES.get(k, k): float(p)
            for k, p in zip(CLASS_KEYS, probabilities)
        }
        st.bar_chart(chart_data)


if __name__ == "__main__":
    main()