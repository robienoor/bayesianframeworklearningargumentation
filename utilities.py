import numpy as np


def convertArgMtrxToAttacks(pos_args,neg_args,arg_mtx):
    all_args = pos_args + neg_args

    attacks = []

    attack_locations = np.where(arg_mtx > 0)
    for attack_location in range(len(attack_locations[0])):
        attack_index = (attack_locations[0][attack_location], attack_locations[1][attack_location])
        attack = (all_args[attack_index[0]], all_args[attack_index[1]])
        attacks.append(attack)

    return attacks

def buildGraph(attacks, pos_args, neg_args):
    noArgs = len(pos_args) + len(neg_args)
    all_args = pos_args + neg_args

    argMtx = np.zeros((noArgs,noArgs))

    if len(attacks) == 0:
        return argMtx
    else:
        for attack in attacks:
            all_args.index(attack[0])
            argMtx[all_args.index(attack[0]),
                all_args.index(attack[1])] = 1

    return argMtx

def getInOutArgs(argMtx):
    sumArgs = argMtx.sum(axis=0)

    inArgs = np.argwhere(sumArgs == 0)
    inArgs = (inArgs.tolist())
    inArgs = [i[0] for i in inArgs]

    attacked = argMtx[inArgs, :]

    outArgs = (np.unique(np.where(attacked>0)[1])).tolist()

    return inArgs, outArgs

def calculateGroundedExtension(argMtx):
    argTypes = np.array(range(0, argMtx.shape[0]))
    ext = []
    terminate = False

    while not terminate:
        inArgs, outArgs = getInOutArgs(argMtx)

        if len(inArgs) > 0:
            ext.extend(list(argTypes[inArgs]))
            argsDelete = inArgs + outArgs
            argMtx = np.delete(argMtx, argsDelete, axis = 0)
            argMtx = np.delete(argMtx, argsDelete, axis = 1)
            argTypes = np.delete(argTypes, argsDelete)

        else:
            break

        sums = np.sum(argMtx.sum(axis=0))

        # If we find that the resulting graph (having deleted current in and out args) is got no more attacks in it then add
        # whatever is leftover to the extension
        if sums == 0:
            ext.extend(list(argTypes))
            terminate = True

    return ext   

