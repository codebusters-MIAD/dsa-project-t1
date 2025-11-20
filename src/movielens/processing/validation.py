"""
Preparacion de targets multi-output para clasificacion de sensibilidad.
"""

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


def prepare_targets(train_df, test_df, config):
    """
    Prepara matrices de targets multi-output para las 5 categorias de sensibilidad.
    Cada categoria tiene 3 niveles: sin_contenido, moderado, alto.
    Retorna matrices Y y diccionario de binarizers.
    """
    target_columns = config['target_columns']
    target_classes = config['target_classes']
    
    mlb_dict = {}
    Y_train_dict = {}
    Y_test_dict = {}
    
    for target_col in target_columns:
        mlb = MultiLabelBinarizer(classes=target_classes)
        
        train_labels = train_df[target_col].astype(str).str.strip().str.lower()
        train_labels = train_labels.replace({"nan": "", "none": "", "": ""})
        train_labels_list = train_labels.apply(
            lambda s: [s] if s in target_classes else []
        )
        
        test_labels = test_df[target_col].astype(str).str.strip().str.lower()
        test_labels = test_labels.replace({"nan": "", "none": "", "": ""})
        test_labels_list = test_labels.apply(
            lambda s: [s] if s in target_classes else []
        )
        
        Y_train_dict[target_col] = mlb.fit_transform(train_labels_list)
        Y_test_dict[target_col] = mlb.transform(test_labels_list)
        mlb_dict[target_col] = mlb
    
    Y_train = np.column_stack([Y_train_dict[col] for col in target_columns])
    Y_test = np.column_stack([Y_test_dict[col] for col in target_columns])
    
    return Y_train, Y_test, mlb_dict
