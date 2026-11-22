import torch
import torch.nn as nn

from model.layers import Mu0Network, Mu1Network, PrpsyNetwork, ShareNetwork, TauNetwork, sigmod2


class PPICFR(nn.Module):

    def __init__(self, prpsy_network: PrpsyNetwork, \
                 mu1_network: Mu1Network, mu0_network: Mu0Network, tau_network: TauNetwork, shareNetwork: ShareNetwork,
                 notshareNetwork: ShareNetwork,
                 next_shareNetwork: ShareNetwork, hs_mu1_network: Mu1Network, hs_mu0_network: Mu0Network,
                 all_mu1_network: Mu1Network, all_mu0_network: Mu0Network, all_prpsy_network: PrpsyNetwork,
                 cfg, device, tarreg=False, freeze_all_prpsy_network=None, share_rep=False, prpsy_shareNetwork: ShareNetwork=None, y_scaler=None):
        super(PPICFR, self).__init__()
        # self.feature_extractor = feature_extractor
        self.shareNetwork = shareNetwork.to(device)
        self.notshareNetwork = notshareNetwork.to(device)
        self.prpsy_network = prpsy_network.to(device)
        self.mu1_network = mu1_network.to(device)
        self.mu0_network = mu0_network.to(device)
        self.tau_network = tau_network.to(device)

        self.next_shareNetwork = next_shareNetwork.to(device)
        self.hs_mu1_network = hs_mu1_network.to(device)
        self.hs_mu0_network = hs_mu0_network.to(device)
        self.all_mu1_network = all_mu1_network.to(device)
        self.all_mu0_network = all_mu0_network.to(device)
        self.all_prpsy_network = all_prpsy_network.to(device)
        if freeze_all_prpsy_network is not None:
            self.freeze_all_prpsy_network = freeze_all_prpsy_network.to(device)
        else:
            self.freeze_all_prpsy_network = self.all_prpsy_network

        if prpsy_shareNetwork is not None:
            print("prpsy_shareNetwork is not None")
            self.prpsy_shareNetwork = prpsy_shareNetwork.to(device)
        else:
            self.prpsy_shareNetwork = self.shareNetwork

        self.cfg = cfg
        self.tarreg = tarreg
        if self.tarreg:
            self.epsilon = nn.Linear(in_features=1, out_features=1)
            self.all_epsilon = nn.Linear(in_features=1, out_features=1)
        self.device = device
        self.share_rep = share_rep
        self.y_scaler = y_scaler
        self.to(device)

    def forward(self, inputs, next_inputs):
        shared_h = self.shareNetwork(inputs)

        p_prpsy_logit_main = self.prpsy_network(shared_h)
        p_prpsy_main = (torch.sigmoid(p_prpsy_logit_main) + 0.01) / 1.02

        next_share_h = self.next_shareNetwork(next_inputs)

        # propensity output_logit
        prpsy_share_h = self.prpsy_shareNetwork(inputs)
        p_prpsy_logit = self.prpsy_network(prpsy_share_h)

        # p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.05, 0.95)
        # p_prpsy = torch.clip(torch.sigmoid(p_prpsy_logit), 0.001, 0.999)
        p_prpsy = (torch.sigmoid(p_prpsy_logit) + 0.01) / 1.02

        # logit for mu1, mu0
        mu1_logit = self.mu1_network(shared_h)
        mu0_logit = self.mu0_network(shared_h)
        # mu1_logit = self.outcome_norm(self.mu1_network(shared_h))
        # mu0_logit = self.outcome_norm(self.mu0_network(shared_h))

        # pseudo tau
        tau_logit = self.tau_network(shared_h)

        p_mu1 = sigmod2(mu1_logit)
        p_mu0 = sigmod2(mu0_logit)
        p_h1 = p_mu1  # Refer to the naming in TARnet/CFR
        p_h0 = p_mu0  # Refer to the naming in TARnet/CFR

        # entire space
        p_estr = torch.mul(p_prpsy, p_h1)
        p_i_prpsy = 1 - p_prpsy
        p_escr = torch.mul(p_i_prpsy, p_h0)

        if self.share_rep:
            final_shared_h = shared_h.detach()
        else:
            final_shared_h = self.notshareNetwork(inputs)

        p_prpsy_logit_hs = self.prpsy_network(shared_h)
        p_prpsy_hs = (torch.sigmoid(p_prpsy_logit_hs) + 0.01) / 1.02

        all_mu1_logit = self.all_mu1_network(torch.cat((final_shared_h, next_share_h), dim=1))
        all_mu0_logit = self.all_mu0_network(torch.cat((final_shared_h, next_share_h), dim=1))
        # all_mu1_logit = self.all_mu1_network(final_shared_h)
        # all_mu0_logit = self.all_mu0_network(final_shared_h)
        hs_mu1_logit = self.hs_mu1_network(next_share_h)
        hs_mu0_logit = self.hs_mu0_network(next_share_h)

        all_p_mu1 = sigmod2(all_mu1_logit)
        all_p_mu0 = sigmod2(all_mu0_logit)
        hs_p_mu1 = sigmod2(hs_mu1_logit)
        hs_p_mu0 = sigmod2(hs_mu0_logit)

        all_prpsy_nograd_logit = self.all_prpsy_network(torch.cat((prpsy_share_h.detach(), next_share_h.detach()), dim=1))
        all_prpsy_logit = self.freeze_all_prpsy_network(torch.cat((prpsy_share_h.detach(), next_share_h), dim=1))

        # all_p_prpsy_nograd = torch.clip(sigmod2(all_prpsy_nograd_logit), 0.001, 0.999)
        # all_p_prpsy = torch.clip(sigmod2(all_prpsy_logit), 0.001, 0.999)
        all_p_prpsy_nograd = (sigmod2(all_prpsy_nograd_logit) + 0.01) / 1.02
        all_p_prpsy = (sigmod2(all_prpsy_logit) + 0.01) / 1.02

        if self.tarreg:
            eps = self.epsilon(torch.ones_like(p_prpsy)[:, 0:1])
            eps_item_1 = 1 / p_prpsy
            mu1_logit_pert = mu1_logit + eps * eps_item_1
            eps_item_0 = - 1 / (1 - p_prpsy)
            mu0_logit_pert = mu0_logit + eps * eps_item_0

            all_eps = self.all_epsilon(torch.ones_like(p_prpsy)[:, 0:1])
            all_eps_item_1 = 1 / p_prpsy
            all_mu1_logit_pert = all_mu1_logit + all_eps * all_eps_item_1
            all_eps_item_0 = - 1 / (1 - p_prpsy)
            all_mu0_logit_pert = all_mu0_logit + all_eps * all_eps_item_0

            return p_prpsy_logit, p_estr, p_escr, tau_logit, mu1_logit, mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h, final_shared_h, \
                all_mu1_logit, all_mu0_logit, hs_mu1_logit, hs_mu0_logit, all_prpsy_nograd_logit, all_prpsy_logit, \
                all_p_mu1, all_p_mu0, hs_p_mu1, hs_p_mu0, all_p_prpsy_nograd, all_p_prpsy, \
                eps, mu1_logit_pert, mu0_logit_pert, all_eps, all_mu1_logit_pert, all_mu0_logit_pert

        return p_prpsy_logit, p_estr, p_escr, tau_logit, mu1_logit, mu0_logit, p_prpsy, p_mu1, p_mu0, p_h1, p_h0, shared_h, final_shared_h, \
            all_mu1_logit, all_mu0_logit, hs_mu1_logit, hs_mu0_logit, all_prpsy_nograd_logit, all_prpsy_logit, \
            all_p_mu1, all_p_mu0, hs_p_mu1, hs_p_mu0, all_p_prpsy_nograd, all_p_prpsy

class PPICFR_on_DRCFR(nn.Module):
    def __init__(self, prpsy_network: PrpsyNetwork, \
                 mu1_network: Mu1Network, mu0_network: Mu0Network, shareNetwork: ShareNetwork,
                 DRCFR_network, next_shareNetwork: ShareNetwork, all_prpsy_network: PrpsyNetwork,
                 hs_mu1_network: Mu1Network, hs_mu0_network: Mu0Network,
                 device, freeze_all_prpsy_network=None, prpsy_shareNetwork: ShareNetwork=None, y_scaler=None):
        super(PPICFR_on_DRCFR, self).__init__()
        self.shareNetwork = shareNetwork.to(device)
        self.prpsy_network = prpsy_network.to(device)
        self.mu1_network = mu1_network.to(device)
        self.mu0_network = mu0_network.to(device)
        self.hs_mu1_network = hs_mu1_network.to(device)
        self.hs_mu0_network = hs_mu0_network.to(device)
        self.DRCFR_network = DRCFR_network.to(device)
        self.next_shareNetwork = next_shareNetwork.to(device)
        self.all_prpsy_network = all_prpsy_network.to(device)
        if freeze_all_prpsy_network is not None:
            self.freeze_all_prpsy_network = freeze_all_prpsy_network.to(device)
        else:
            self.freeze_all_prpsy_network = self.all_prpsy_network

        if prpsy_shareNetwork is not None:
            print("prpsy_shareNetwork is not None")
            self.prpsy_shareNetwork = prpsy_shareNetwork.to(device)
        else:
            self.prpsy_shareNetwork = self.shareNetwork
        self.tarreg = False
        self.device = device
        self.y_scaler = y_scaler

    def forward(self, inputs, next_inputs):
        shared_h = self.shareNetwork(inputs)

        next_share_h = self.next_shareNetwork(next_inputs)

        prpsy_share_h = self.prpsy_shareNetwork(inputs)
        p_prpsy_logit = self.prpsy_network(prpsy_share_h)
        p_prpsy = (torch.sigmoid(p_prpsy_logit) + 0.01) / 1.02

        hs_mu1_logit = self.hs_mu1_network(next_share_h)
        hs_mu0_logit = self.hs_mu0_network(next_share_h)
        hs_p_mu1 = sigmod2(hs_mu1_logit)
        hs_p_mu0 = sigmod2(hs_mu0_logit)

        p_prpsy_logit_hs, p_prpsy_hs, all_mu1_logit, all_mu0_logit, all_p_tau, \
        all_p_yf, all_p_ycf, final_gamma, final_delta, final_upsilon = self.DRCFR_network(inputs, next_share_h)

        p_prpsy_logit_main = self.prpsy_network(final_delta + final_upsilon)
        p_prpsy_main = (torch.sigmoid(p_prpsy_logit_main) + 0.01) / 1.02

        mu1_logit = self.mu1_network(final_gamma + final_delta)
        mu0_logit = self.mu0_network(final_gamma + final_delta)

        p_mu1 = sigmod2(mu1_logit)
        p_mu0 = sigmod2(mu0_logit)
        p_h1 = p_mu1
        p_h0 = p_mu0

        all_prpsy_nograd_logit = self.all_prpsy_network(torch.cat((prpsy_share_h.detach(), next_share_h.detach()), dim=1))
        all_prpsy_logit = self.freeze_all_prpsy_network(torch.cat((prpsy_share_h.detach(), next_share_h), dim=1))

        all_p_prpsy_nograd = (sigmod2(all_prpsy_nograd_logit) + 0.01) / 1.02
        all_p_prpsy = (sigmod2(all_prpsy_logit) + 0.01) / 1.02

        return p_prpsy_logit, p_prpsy, p_prpsy_logit_main, p_prpsy_main, p_prpsy_logit_hs, p_prpsy_hs, \
            shared_h, final_gamma, final_delta, final_upsilon, mu1_logit, mu0_logit, p_mu1, p_mu0, p_h1, p_h0, \
            all_mu1_logit, all_mu0_logit, all_prpsy_nograd_logit, all_prpsy_logit, all_p_prpsy_nograd, all_p_prpsy, \
            hs_mu1_logit, hs_mu0_logit, hs_p_mu1, hs_p_mu0
