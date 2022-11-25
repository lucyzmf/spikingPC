# %%
###############################################################################################
##########################                  FTTP                ###############################
###############################################################################################

# fptt parameters
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

alpha = .5
beta = .5
rho = 0.
# %%
def get_stats_named_params( model ):
    named_params = {}
    for name, param in model.named_parameters():
        sm, lm, dm = param.detach().clone(), 0.0*param.detach().clone(), 0.0*param.detach().clone()
        named_params[name] = (param, sm, lm, dm)
    return named_params

def post_optimizer_updates( named_params):
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        lm.data.add_( -alpha * (param - sm) )
        sm.data.mul_( (1.0-beta) )
        sm.data.add_( beta * param - (beta/alpha) * lm )

def get_regularizer_named_params( named_params,  _lambda=1.0 ):
    regularization = torch.zeros( [], device=device )
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        regularization += (rho-1.) * torch.sum( param * lm )
        r_p = _lambda * 0.5 * alpha * torch.sum( torch.square(param - sm) )
        regularization += r_p
        # print(name,r_p)
    return regularization 

def reset_named_params(named_params):
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        param.data.copy_(sm.data)