import copy


# Clase para la parada temprana durante el entrenamiento
class early_stopper(object):
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Inicializa el objeto de parada temprana.
        :param patience: número máximo de rondas toleradas
        :param verbose: indicador para imprimir información durante el entrenamiento
        :param delta: factor de regularización
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_value = None
        self.best_cv = None
        self.is_earlystop = False
        self.count = 0
        self.best_model = None

    def earlystop(self, loss, model=None):
        """
        Función para la parada temprana.
        :param loss: puntuación de pérdida en el conjunto de validación
        :param model: el modelo
        """
        value = -loss
        cv = loss

        if self.best_value is None:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to('cpu')
        elif value < self.best_value + self.delta:
            self.count += 1
            if self.verbose:
                print('EarlyStop count: {:02d}'.format(self.count))
            if self.count >= self.patience:
                self.is_earlystop = True
        else:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to('cpu')
            self.count = 0
