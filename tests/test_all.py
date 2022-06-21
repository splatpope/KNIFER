import knifer.cli as CLI
import knifer.context as KF
import knifer.config.params as P

if __name__ == "__main__":
    ARGS = CLI.init()
    ex = CLI.create_experiment(ARGS)
    tp = P.TrainingParameters(50, 5)
