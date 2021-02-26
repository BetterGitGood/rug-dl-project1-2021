import matplotlib.pyplot as plt
import pandas as pd

legend = True

measures = ['Loss', 'Accuracy']
runs = ['First', 'second']
for measure in measures:
    for run in runs:
        for modelnr in range(0,5):
            run_adam = pd.read_csv('./'+run+' run/' + 'running-results'+ str(modelnr) + 'Adam')
            run_adam_w = pd.read_csv('./'+run+' run/' + 'running-results'+ str(modelnr) + 'AdamW')
            
            if modelnr == 0:
                label_gen = 'ResNet18'
            elif modelnr == 1:
                label_gen = 'ResNet34'
            elif modelnr == 2:
                label_gen = 'ResNet50'
            elif modelnr == 3:
                label_gen = 'ResNet101'
            else:
                label_gen = 'Naive Student Network'
            
            if run == 'second':
                plt.plot(run_adam.index, run_adam[measure],'--', label=label_gen + ' Adam')
                plt.plot(run_adam_w.index, run_adam_w[measure],'--', label=label_gen + ' AdamW')
            else:
                plt.plot(run_adam.index, run_adam[measure], label=label_gen + ' Adam')
                plt.plot(run_adam_w.index, run_adam_w[measure], label=label_gen + ' AdamW')
            
    plt.title('Training ' + measure)    
    plt.xlabel('Iterations x2000')
    plt.ylabel(measure)
    if legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig('Training_' + measure + '_' + str(legend), bbox_inches='tight')
    plt.clf()

print("End")