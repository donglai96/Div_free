# -*- coding: utf-8 -*-
import operalib as ovk

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from numpy.random import RandomState

from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge


def main():
    """Example of vector-field learning."""

    # Fix a seed
    random_state = RandomState(0)

    # Generate data
    inputs, targets = ovk.toy_data_div_free_field(n_samples=500)
    print(inputs)
    print(targets)
    inputs_mesh = ovk.array2mesh(inputs)

    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, train_size=(inputs.shape[0] - 50),
        random_state=random_state)
    print()
    #Add some noise
    targets_train = (targets_train +
                     .175 * random_state.randn(targets_train.shape[0],
                                               targets_train.shape[1]))

    regressor = {'DF':
                 ovk.OVKRidge(ovkernel=ovk.RBFDivFreeKernel(gamma=5.),
                              lbda=1e-4),
                 'Indep':
                 KernelRidge(kernel='rbf', gamma=.5, alpha=1e-4)}

    # Learning with div-free
    regressor['DF'].fit(inputs_train, targets_train)
    score_cf = regressor['DF'].score(inputs_test, targets_test)
    print('R2 div-free ridge: %.5f' % score_cf)
    targets_mesh_cf = ovk.array2mesh(regressor['DF'].predict(inputs))

    # Learning with sklearn ridge
    regressor['Indep'].fit(inputs_train, targets_train)
    scode_id = regressor['Indep'].score(inputs_test, targets_test)
    print('R2 independent ridge: %.5f' % scode_id)
    print(inputs_test.shape)
    print("test data")
    print(inputs_test.shape)
    print(type(inputs_test))

    targets_mesh_id = ovk.array2mesh(regressor['Indep'].predict(inputs))

    # Plotting
    # pylint: disable=E1101
    fig, axarr = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14, 7))
    axarr[0].streamplot(inputs_mesh[0], inputs_mesh[1],
                        targets_mesh_cf[0], targets_mesh_cf[1],
                        color=np.sqrt(targets_mesh_cf[0]**2 +
                                      targets_mesh_cf[1]**2),
                        linewidth=.5, cmap=plt.cm.jet, density=2,
                        arrowstyle=u'->')
    axarr[1].streamplot(inputs_mesh[0], inputs_mesh[1],
                        targets_mesh_id[0], targets_mesh_id[1],
                        color=np.sqrt(targets_mesh_id[0]**2 +
                                      targets_mesh_id[1]**2),
                        linewidth=.5, cmap=plt.cm.jet, density=2,
                        arrowstyle=u'->')
    axarr[0].set_ylim([-1, 1])
    axarr[0].set_xlim([-1, 1])
    axarr[0].set_title('div-Free Ridge, R2: %.5f' % score_cf)
    axarr[1].set_ylim([-1, 1])
    axarr[1].set_xlim([-1, 1])
    axarr[1].set_title('Independent Ridge, R2: %.5f' % scode_id)

    fig.suptitle('Vectorfield learning')
    plt.show()


if __name__ == '__main__':
    main()
