import torch
from torch.nn import functional as F
from torch import Tensor
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from cleverhans.future.torch.attacks.fast_gradient_method import *
from cleverhans.future.torch.attacks.noise import *
from one_pixel_attack import OnePixelAttack
import utilities as ut
import numpy as np
import matplotlib.pyplot as plt
import statistics
import os


'''
Functions for evaluating kl metrics on mnist set
'''

def evaluate_attack_mnist(model,
                          device,
                          attack,
                          eps=0,
                          norm=2,
                          num_SD=1.8,
                          num_summed_SD=0.75,
                          num_false = 1,
                          num_imgs=10000,
                          print_every=100,
                          stop_idx=[],
                          CheckAll=False,
                          use_printouts=False,
                          get_psi=False
                          ):

    '''

    :param model (nn.Module): The model to be used for testing. It is expected to have the following attributes:
        .name (str): Name of the model
        .encoder (nn.Module): Neural network with callable, implemented forward function -- must expect inputs of size 784
        .classifier (nn.Module): Neural network with callable, implemented forward function -- must expect inputs of size 784
        .z_prior (torch.nn.Parameter): length-2 tuple that holds the z_prior mean(s) in [0] and z_prior variances in [1]
    :param device (str): either 'cuda' or 'cpu'
    :param attack (str): Choice of 'noise', 'fgm', or 'pixel'
    :param eps (int or float): Strength of attack. Note that pixel attacks only takes integer values
    :param norm (int or float): Either 2 or np.inf. Used only with fgm attacks
    :param num_SD (float): sigma_detect threshold value
    :param num_summed_SD (float): Sigma_detect threshold value
    :param num_false (int): Number of delta values for a given image that must exceed sigma_detect to be detected
    :param num_imgs (int): Number of images to iterate over through the test dataset
    :param print_every (int): How often to print a progress report
    :param stop_idx (list of ints): List of specific indexes within the test dataset to pause at with detailed printouts
    :param CheckAll (bool): If true, will pause at every image within the test dataset
    :param use_printouts (bool): If true, will pause at every anomalous / successful adversarial image in the dataset
    :param get_psi (bool): If true (and eps = 0), will evaluate and save psi values across the dataset
            as 'psis_minst.npy'
    '''

    # Load MNIST test dataset
    testload = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data/mnist",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            )
        )
    )
    # x_test = testload.dataset.test_data.to(device).reshape(-1, 784).float() / 255
    x_test = testload.dataset.test_data.to(device).reshape(-1, 784).float()[:num_imgs] / 255
    y_test = testload.dataset.test_labels.to(device)[:num_imgs]
    print("y_test shape: {}".format(y_test.shape))
    total_num = len(x_test)
    print("Total length of x_test dataset: {}".format(total_num))

    # Load model in eval mode
    model.eval()

    # Load KL Data
    KL_Classes_Stats = np.zeros((10, 10, 2))
    if os.path.exists('deltas_mnist.npy'):
        KL_Classes_Stats = np.load('deltas_mnist.npy')
    else:
        print("Warning: No deltas_mnist file to load. Make sure you run determine_deltas_mnist first!")
    KL_Summed_SD_Stats = np.zeros((10, 2))
    if os.path.exists('psis_mnist.npy'):
        KL_Summed_SD_Stats = np.load('psis_mnist.npy')
    else:
        print("Warning: No psis_mnist file to load. Make sure you run with get_psi=True first!")

    # Create vectors to hold values
    Max_Delta_KL_z_adv = []
    Summed_KL_z_adv = []
    PredictClean = []
    ProbClean = []
    ProbAdv = []
    PredictAdv = []
    IsCorrect = []
    AdvImages = []
    SuccessfulAdvAtkDetected = []
    UnsuccessfulAdvAtkDetected = []
    FalsePositive = []      # only used for d = 0 pixels changed
    AnomalyDetected = []
    KL_Summed_SD = []
    for i in range(10):
        KL_Summed_SD.append([])

    # If running Single Pixel attack, load class
    attacker = None
    if attack == 'pixel':
        attacker = OnePixelAttack(model, device)



    for x, y, j in zip(x_test, y_test, range(len(x_test))):
        # Load single img
        orig = x.view(28, 28).cpu().numpy()
        img = orig.copy()
        shape = img.shape

        inp = Variable(x.type(Tensor)).to(device)
        prob_orig = ut.softmax(model.classifier(inp).data.cpu().numpy()[0])
        pred_orig = np.argmax(prob_orig)

        # Append to vectors
        PredictClean.append(pred_orig)
        ProbClean.append(prob_orig)

        # Run specified attack
        adv_img = None
        if eps > 0:
            if attack == 'fgm':
                adv_img = fast_gradient_method(model.classifier, x, eps=eps, norm=norm, clip_min=0, clip_max=1).view(1,-1)
            elif attack == 'noise':
                adv_img = noise(x, eps=eps, clip_min=0, clip_max=1).view(1,-1)
            elif attack == 'pixel':
                _, _, _, adv_img = attacker.pixel_attack(eps, shape, pred_orig, img)
            else:
                raise AssertionError("Attack must either be 'fgm', 'pixel', or 'noise'")
        else:
            adv_img = x.view(1,-1)
        adv_out = model.classifier(adv_img)
        prob = ut.softmax(adv_out.data.cpu().numpy())
        adv_y = F.softmax(adv_out, dim=-1).float()
        pred_adv = torch.topk(adv_y, 1, dim=-1)[1].item()
        prob_adv = prob[0][pred_adv]

        # Append to vectors
        PredictAdv.append(pred_adv)
        ProbAdv.append(prob_adv)
        AdvImages.append(adv_img.view(1, 28, 28).data)

        # Append to accuracy vector
        IsCorrect.append(int(pred_adv == y))

        #### Test KL z div for all images ####

        # Display adv image only if certain conditions are met
        if(((pred_orig != pred_adv) or (pred_orig != y) or CheckAll) and use_printouts) or j in stop_idx:
            fig1 = plt.imshow(adv_img.view(28, 28).cpu().data)
            fig1.axes.get_xaxis().set_visible(False)
            fig1.axes.get_yaxis().set_visible(False)
            plt.title(
                '{} Attack, eps = {}, Adv Prediction: {}'.format(
                    attack, eps, pred_adv))
            plt.show()
            fig2 = plt.imshow(x.view(28, 28).cpu().data)
            fig2.axes.get_xaxis().set_visible(False)
            fig2.axes.get_yaxis().set_visible(False)
            plt.title('Clean Image Prediction: {}'.format(pred_orig))
            plt.show()
        if(((pred_orig != pred_adv) or (pred_orig != y) or CheckAll) and use_printouts) or j in stop_idx:
            print("Test Image i = {}: Original prediction: {}, Adversarially-induced prediction: {}, True Label = {}".format(
            j, pred_orig, pred_adv, y))
        KL_local = []

        # Calculate KL div for "expected" (clean or adversarially-induced) label
        y_prob = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).float().view(1, -1).to(device)
        y_prob[0][pred_adv] = 1
        qm, qv = model.encoder(adv_img.view(1, -1), y_prob)
        kl_z_all = y_prob * ut.kl_normal(qm, qv, model.z_prior[0],
                                         model.z_prior[1])  # kl_z_all shape = [batch_size * y_dim]
        expected_kl_z = torch.sum(kl_z_all)
        TotalFalse = 0
        Num_SD_Away_Total = 0
        Max_Adv_KL = 0
        for i in range(10):
            y_prob = torch.tensor([0,0,0,0,0,0,0,0,0,0]).float().view(1,-1).to(device)
            y_prob[0][i] = 1
            #y_logprob = F.log_softmax(out, dim=1).float()
            qm, qv = model.encoder(adv_img.view(1,-1), y_prob)
            kl_z_all = y_prob * ut.kl_normal(qm, qv, model.z_prior[0],
                                             model.z_prior[1])  # kl_z_all shape = [batch_size * y_dim]
            kl_z = torch.sum(kl_z_all)
            KL_local.append(kl_z.item())
            if KL_Classes_Stats[pred_adv][i][1] > 0:
                Num_SD_Away = (abs(kl_z - expected_kl_z - KL_Classes_Stats[pred_adv][i][0]) / KL_Classes_Stats[pred_adv][i][1]).item()
            else:
                Num_SD_Away = 0
            if Num_SD_Away > Max_Adv_KL:
                Max_Adv_KL = Num_SD_Away
            Num_SD_Away_Total = Num_SD_Away_Total + Num_SD_Away
            reasonable = True if Num_SD_Away <= num_SD else False
            if not reasonable:
                TotalFalse = TotalFalse + 1
            if(((pred_orig != pred_adv) or (pred_orig != y) or CheckAll) and use_printouts) or j in stop_idx:
                print("delta KL_div for y = {}: {:.2f}, Expected delta KL_div: {:.2f}, SD: {:.2f}, Num SD Away: {:.2f}, Reasonable (within {} SD): {}".format(
                    i, kl_z - expected_kl_z, KL_Classes_Stats[pred_adv][i][0], KL_Classes_Stats[pred_adv][i][1],
                    Num_SD_Away,
                    num_SD, reasonable))

        PositiveDetected = 1 if (Num_SD_Away_Total - KL_Summed_SD_Stats[pred_adv][0]) / KL_Summed_SD_Stats[pred_adv][1] > num_summed_SD else 0

        if(pred_orig != pred_adv) or (eps == 0 and pred_orig != y):
            Max_Delta_KL_z_adv.append(Max_Adv_KL)
            Summed_KL_z_adv.append(Num_SD_Away_Total)

        if eps == 0 and get_psi:
            KL_Summed_SD[y].append(Num_SD_Away_Total.item())

        if(((pred_orig != pred_adv) or (pred_orig != y) or CheckAll) and use_printouts) or j in stop_idx:
            print("Summed SDs across classes: {:.2f}".format(Num_SD_Away_Total))
            print("Mean, SD for Summed SDs: {}".format(KL_Summed_SD_Stats[pred_adv]))
            print("Detected: {}, PositiveDetected: {}, Detected as anomaly: {}".format(TotalFalse >= num_false, PositiveDetected, bool(TotalFalse >= num_false or PositiveDetected)))

        # Append the Detected Value to the appropriate vector
        if eps == 0 and pred_orig == y:  # Then this is a false positive
            FalsePositive.append(int(TotalFalse >= num_false or PositiveDetected))
        if pred_orig == pred_adv and TotalFalse >= num_false:  # Then this is a detection of an unsuccessful adv atk
            UnsuccessfulAdvAtkDetected.append(PositiveDetected)
        if pred_orig != pred_adv and pred_orig == y:  # Then this is a detection of a successful adv atk
            SuccessfulAdvAtkDetected.append(int(TotalFalse >= num_false or PositiveDetected))
        if eps == 0 and pred_orig != y:               # Then this is a detection of anomaly
            AnomalyDetected.append(int(TotalFalse >= num_false or PositiveDetected))

        # Wait for user to press a keystroke before continuing
        if(((pred_orig != pred_adv) or (pred_orig != y) or CheckAll) and use_printouts) or j in stop_idx:
            input("Press Enter to continue...")

        # progress print
        if j and j % print_every == 0:
            # Get ongoing stats printed out
            Accuracy = statistics.mean(IsCorrect) * 100
            Avg_Max_Delta_KL_z_adv = statistics.mean(Max_Delta_KL_z_adv)
            SD_Max_Delta_KL_z_adv = statistics.stdev(Max_Delta_KL_z_adv)
            Avg_Summed_KL_z_adv = statistics.mean(Summed_KL_z_adv)
            SD_Summed_KL_z_adv = statistics.stdev(Summed_KL_z_adv)
            print("Completed {} of {} Total Examples in MNIST Test Dataset. "
                  "Accuracy = {:.2f}, "
                  "Avg Max Delta Adversarial KL_z = {:.2f}, SD = {:.2f}, "
                  "Avg Summed Delta Adversarial KL_z = {:.2f}, SD = {:.2f}".format(
                    j, total_num, Accuracy, Avg_Max_Delta_KL_z_adv, SD_Max_Delta_KL_z_adv,
                Avg_Summed_KL_z_adv, SD_Summed_KL_z_adv))

    # After, determine stats
    Accuracy = statistics.mean(IsCorrect) * 100
    Avg_Max_Delta_KL_z_adv = statistics.mean(Max_Delta_KL_z_adv)
    SD_Max_Delta_KL_z_adv = statistics.stdev(Max_Delta_KL_z_adv)
    Avg_Summed_KL_z_adv = statistics.mean(Summed_KL_z_adv)
    SD_Summed_KL_z_adv = statistics.stdev(Summed_KL_z_adv)

    if eps == 0 and get_psi:
        KL_Summed_SD_Stats = np.zeros([10,2])
        for i in range(10):
            KL_Summed_SD_Stats[i][0] = statistics.mean(KL_Summed_SD[i])
            KL_Summed_SD_Stats[i][1] = statistics.stdev(KL_Summed_SD[i])
        # Save file
        np.save('psis_mnist.npy', KL_Summed_SD_Stats)


    FalsePositivePercentage = None
    SuccessfulAdvAtkDetectedPercentage = None
    AnomalyDetectedPercentage = None
    if eps == 0 and len(FalsePositive) > 0:
        FalsePositivePercentage = sum(FalsePositive) / len(x_test) * 100
    if len(SuccessfulAdvAtkDetected) > 0:
        SuccessfulAdvAtkDetectedPercentage = statistics.mean(SuccessfulAdvAtkDetected) * 100
    if len(AnomalyDetected) > 0:
        AnomalyDetectedPercentage = statistics.mean(AnomalyDetected) * 100

    # Print out results to user
    print("Accuracy with eps = {} {} Disturbance: {:.2f}%".format(eps, attack, Accuracy))
    print("Percentage of Successful Adversarial Attacks: {:.2f}%".format(100*len(SuccessfulAdvAtkDetected)/len(x_test)))
    print("Average Max Delta Adversarial KL_z = {:.2f}, SD = {:.2f}".format(Avg_Max_Delta_KL_z_adv, SD_Max_Delta_KL_z_adv))
    print("Average Summed Delta Adversarial KL_z = {:.2f}, SD = {:.2f}".format(Avg_Summed_KL_z_adv, SD_Summed_KL_z_adv))
    if eps == 0:
        print("False Positive Percentage for Clean (eps = {}) data with KL threshold of {}: {}%".format(eps, num_SD, FalsePositivePercentage))
        print("Anomaly (incorrectly classified from clean img) Detected Percentage: {:.2f}%".format(
            AnomalyDetectedPercentage))
    else:
        print("Successful Adversarial Attack Detected Percentage: {:.2f}%".format(SuccessfulAdvAtkDetectedPercentage))

    # Now, plot the histograms of the KL divergences of both the clean and corrupted images separately
    plt.figure(0)
    plt.hist(x=Max_Delta_KL_z_adv, bins='auto', color='#0504aa')
    plt.grid(axis='y')
    plt.xlabel('Max KL z Divergence')
    plt.ylabel('Frequency')
    plt.xlim(0, 5)
    if eps == 0:
        plt.title("Max Clean Delta using {} Model on MNIST".format(model.name))
    else:
        plt.title('Max Adv. Delta using {} Model on MNIST, {} Attack, eps = {}'.format(model.name, attack, eps))

    plt.show()

    # Now, plot the histograms of the KL divergences of both the clean and corrupted images separately
    plt.figure(1)
    plt.hist(x=Summed_KL_z_adv, bins='auto', color='#607c8e')
    plt.grid(axis='y')
    plt.xlabel('Summed KL z Divergence')
    plt.ylabel('Frequency')
    plt.xlim(0, 35)
    if eps == 0:
        plt.title("Clean Psi using {} on MNIST".format(model.name))
    else:
        plt.title('Adv. Psi using {} on MNIST, {} Attack, eps = {}'.format(model.name, attack, eps))
    plt.show()

    # Save some of the examples of Adv images generated
    save_image(AdvImages[:25], "images/{}_attack-eps={}.png".format(attack, eps), nrow=5, normalize=True)


def determine_deltas_mnist(model, device):
    '''
    Calculates the individual relative expected (mean, stddev) KLs per class, given clean images
    Meant to be used for anomaly detection for adversarial attacks. Saves resulting (Delta Bars and sigmas) in file
    as 'deltas_mnist.npy'

    :param model (nn.Module): The model to be used for testing. It is expected to have the following attributes:
        .name (str): Name of the model
        .encoder (nn.Module): Neural network with callable, implemented forward function
        .classifier (nn.Module): Neural network with callable, implemented forward function
        .z_prior (torch.nn.Parameter): length-2 tuple that holds the z_prior mean(s) in [0] and z_prior variances in [1]
    :param device (str): either 'cuda' or 'cpu'
    '''


    # Load MNIST test dataset
    testload = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data/mnist",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            )
        )
    )
    x_test = testload.dataset.test_data.to(device).reshape(-1, 784).float() / 255
    y_test = testload.dataset.test_labels.to(device)

    # Load model in eval mode
    model.eval()


    # Create vectors to hold values
    [[],[],[],[],[],[],[],[],[],[]]
    dummy = []
    for i in range(10):
        dummy.append([])
    KL_Class0 = [[],[],[],[],[],[],[],[],[],[]]                # 2D array to hold KL values for each other (incorrect) class - diff between true and other KL value
    KL_Class1 = [[],[],[],[],[],[],[],[],[],[]]
    KL_Class2 = [[],[],[],[],[],[],[],[],[],[]]
    KL_Class3 = [[],[],[],[],[],[],[],[],[],[]]
    KL_Class4 = [[],[],[],[],[],[],[],[],[],[]]
    KL_Class5 = [[],[],[],[],[],[],[],[],[],[]]
    KL_Class6 = [[],[],[],[],[],[],[],[],[],[]]
    KL_Class7 = [[],[],[],[],[],[],[],[],[],[]]
    KL_Class8 = [[],[],[],[],[],[],[],[],[],[]]
    KL_Class9 = [[],[],[],[],[],[],[],[],[],[]]
    KL_Classes = [KL_Class0, KL_Class1, KL_Class2, KL_Class3, KL_Class4, KL_Class5, KL_Class6, KL_Class7, KL_Class8,
                  KL_Class9]

    for x, y, j in zip(x_test, y_test, range(len(x_test))):
        # Load single img
        orig = x.view(28, 28).cpu().numpy()
        img = orig.copy()
        img = Variable(x.type(Tensor)).to(device)

        # Calculate KL div for true label
        y_prob = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).float().view(1, -1).to(device)
        y_prob[0][y] = 1
        qm, qv = model.enc.encode(img.view(1, -1), y_prob)
        kl_z_all = y_prob * ut.kl_normal(qm, qv, model.z_prior[0],
                                         model.z_prior[1])  # kl_z_all shape = [batch_size * y_dim]
        true_kl_z = torch.sum(kl_z_all)


        for i in range(10):
            y_prob = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).float().view(1, -1).to(device)
            y_prob[0][i] = 1
            qm, qv = model.enc.encode(img.view(1, -1), y_prob)
            kl_z_all = y_prob * ut.kl_normal(qm, qv, model.z_prior[0],
                                             model.z_prior[1])  # kl_z_all shape = [batch_size * y_dim]
            kl_z = torch.sum(kl_z_all)
            KL_Classes[y][i].append(kl_z.item() - true_kl_z.item())

    # Save KL_Classes and find stats on KL_Classes

    # Determine means and STDDEV
    KL_Classes_Stats = np.zeros([10,10,2])
    for i in range(10):
        for k in range(10):
            KL_Classes_Stats[i, k, 0] = statistics.mean(KL_Classes[i][k])
            KL_Classes_Stats[i, k, 1] = statistics.stdev(KL_Classes[i][k])

    # Print out excerpt
    for y_label in range(10):
        print("KL_Classes_Stats for Class y = {}: {}".format(y_label, KL_Classes_Stats[y_label, :, :]))

    # Save files
    np.save('deltas_mnist.npy', KL_Classes_Stats)