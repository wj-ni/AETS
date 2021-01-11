from .aets.set import train_set
from .aets.multiset import train_set
from .aets.sequence import train_sequence
def train(file, ts_type, time_unit):
    if ts_type == 'set':
        MAE, MSE = train_set(file, ts_type, time_unit)
    elif ts_type == 'multiset':
        MAE, MSE = train_multiset(file, ts_type, time_unit)
    else:
        MAE, MSE = train_sequence(file, ts_type, time_unit)

    print("totalMSE:," + str(MSE) + "\n")
    print("totalMAE:," + str(MAE) + "\n")


if __name__ == '__main__':
    path = "./data/"
    file = "helpdesk.csv"
    typeList = ['set']
    for i in range(10):
        train(file, 'set', 'day')
