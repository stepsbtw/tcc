import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# --- CONFIGURAÇÕES BÁSICAS ---
DATASET_DIR = Path("IPqM-Fall/windowed_synchronized")
CHECKPOINT_DIR = Path("checkpoints_cnn")  # Pasta onde seus .pth e .npy estão salvos
CHEST_DIR = CHECKPOINT_DIR / "CHEST"
LEFT_DIR = CHECKPOINT_DIR / "LEFT"
RIGHT_DIR = CHECKPOINT_DIR / "RIGHT"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
FEATURE_COLUMNS = ["ax", "ay", "az", "amag", "wx", "wy", "wz", "wmag"]
WINDOW_SAMPLES = 180
NUM_CLASSES = 2
DROPOUT = 0.35

# --- DEFINIÇÃO DO MODELO (Necessário para carregar os pesos) ---
class CNN1Conv(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(DROPOUT)
        )
        conv_out_length = WINDOW_SAMPLES + 2*2 - 4 + 1
        pool_out_length = conv_out_length // 3
        flattened_size = 64 * pool_out_length
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, NUM_CLASSES)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def extrair_probabilidade_oof(sensor_name, X_test_raw, test_subject, fold_number, save_dir):
    """
    Carrega o modelo salvo e o normalizador salvo daquele fold específico.
    Retorna apenas as probabilidades geradas para o paciente de teste.
    """
    # 1. Carregar os Normalizadores exatos daquele fold
    mean_path = save_dir / f"{sensor_name}_fold_{fold_number}_mean.npy"
    std_path = save_dir / f"{sensor_name}_fold_{fold_number}_std.npy"
    
    train_mean = np.load(mean_path)
    train_std = np.load(std_path)
    
    # Normalizar o X_test exatamente como o modelo aprendeu
    X_test_scaled = (X_test_raw - train_mean) / train_std
    
    # 2. Preparar dados para o PyTorch
    X_tensor = torch.tensor(X_test_scaled.transpose(0, 2, 1), dtype=torch.float32).to(DEVICE)
    
    # 3. Carregar o Modelo Salvo (O Cérebro Congelado)
    ckpt_path = save_dir / f"{sensor_name}_fold_{fold_number}_subject_{test_subject}.pth"
    model = CNN1Conv(len(FEATURE_COLUMNS)).to(DEVICE)
    
    # Lida com DataParallel caso você tenha treinado com múltiplas GPUs
    estado_salvo = torch.load(ckpt_path, map_location=DEVICE)
    if "module." in list(estado_salvo.keys())[0]:
        model = nn.DataParallel(model)
    
    model.load_state_dict(estado_salvo)
    model.eval() # Modo de inferência (desliga dropout)
    
    # 4. Fazer a Previsão
    probabilidades = []
    with torch.no_grad():
        # Processar em lotes para não estourar a RAM da GPU
        for i in range(0, len(X_tensor), BATCH_SIZE):
            batch = X_tensor[i:i+BATCH_SIZE]
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)
            # Pegar a probabilidade da classe 1 (Queda) e jogar para a CPU
            probabilidades.extend(probs[:, 1].cpu().numpy())
            
    return np.array(probabilidades)

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    return {"accuracy": float(acc), "f1": float(f1)}

if __name__ == "__main__":
    print("--- INICIANDO CONSTRUÇÃO DO DATASET OOF (INFERÊNCIA APENAS) ---")
    
    # Carregar Dados Brutos
    X_chest = np.load(DATASET_DIR / "X_chest.npy")
    X_left = np.load(DATASET_DIR / "X_left.npy")
    X_right = np.load(DATASET_DIR / "X_right.npy")
    y = np.load(DATASET_DIR / "y.npy")
    groups = np.load(DATASET_DIR / "groups.npy")

    logo = LeaveOneGroupOut()
    folds = list(logo.split(X_chest, y, groups))

    # Matrizes vazias para o Dataset do Meta-Modelo
    oof_chest = np.zeros(len(y))
    oof_left = np.zeros(len(y))
    oof_right = np.zeros(len(y))

    pbar = tqdm(folds, total=len(folds), desc="Gerando Previsões OOF")

    for fold_number, (_, test_idx) in enumerate(pbar, start=1):
        test_subject = groups[test_idx][0]
        
        # Pega as probabilidades puras do Sujeito de Teste usando o modelo salvo daquele fold
        probs_chest = extrair_probabilidade_oof("CHEST", X_chest[test_idx], test_subject, fold_number, CHEST_DIR)
        probs_left  = extrair_probabilidade_oof("LEFT", X_left[test_idx], test_subject, fold_number, LEFT_DIR)
        probs_right = extrair_probabilidade_oof("RIGHT", X_right[test_idx], test_subject, fold_number, RIGHT_DIR)
        
        # Salva nos índices corretos
        oof_chest[test_idx] = probs_chest
        oof_left[test_idx] = probs_left
        oof_right[test_idx] = probs_right

    print("\n--- TREINANDO O META-MODELO (STACKING) ---")
    
    # Monta a Matriz do Chefe (Shape: Total_Amostras x 3)
    X_meta = np.column_stack((oof_chest, oof_left, oof_right))
    
    # O Chefe: Regressão Logística
    meta_model = LogisticRegression(class_weight='balanced', random_state=42)
    meta_model.fit(X_meta, y)
    
    # Previsão Final do Meta-Modelo para fins de métrica
    y_pred_meta = meta_model.predict(X_meta)
    metricas = compute_metrics(y, y_pred_meta)
    
    # Visualizando a inteligência que ele aprendeu
    pesos = meta_model.coef_[0]
    print("\n[PESOS APRENDIDOS]")
    print(f"Peito:          {pesos[0]:.4f}")
    print(f"Braço Esquerdo: {pesos[1]:.4f}")
    print(f"Braço Direito:  {pesos[2]:.4f}")
    
    print("\n[DESEMPENHO DO SISTEMA EM CONJUNTO]")
    print(f"F1-Score Final: {metricas['f1']:.4f}")
    print(f"Acurácia Final: {metricas['accuracy']:.4f}")
    
    # Salvar o modelo final
    joblib.dump(meta_model, CHECKPOINT_DIR / "meta_model_stacking.pkl")
    print(f"\nMeta-Modelo salvo em: {CHECKPOINT_DIR / 'meta_model_stacking.pkl'}")
    print("O sistema está pronto para produção!")