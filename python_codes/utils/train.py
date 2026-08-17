"""
train.py -- the single shared training loop used by every runner.

One optimiser everywhere, so all figures are comparable: Adam with a cosine-decayed
learning rate (lr -> alpha*lr over n_epochs) and global-norm clipping. The trainer
takes the target as a runtime argument (compiles once per circuit shape, reused across
targets and entangler families) and returns BOTH the final parameters and the FULL
per-epoch loss history, so convergence is always auditable.

    trainer = make_trainer(forward, n_epochs, lr, clip)
    params, losses = trainer(gates_batch, params0_batch, target)   # vmapped over seeds
        params  : (S, L, 3, N)
        losses  : (S, n_epochs)   NLL at every epoch;  KLD = NLL - H(target)
"""
import jax
import jax.numpy as jnp
import optax

__all__ = ["make_trainer"]


def make_trainer(forward, n_epochs, lr=0.02, clip=1.0, eps=1e-12, alpha=0.01):
    """Return a jitted, vmapped trainer returning (final_params, per-epoch loss history).

    lr is cosine-decayed from `lr` to `alpha*lr` over `n_epochs` steps.
    """
    schedule = optax.cosine_decay_schedule(lr, n_epochs, alpha=alpha)
    opt = optax.chain(optax.clip_by_global_norm(clip), optax.adam(schedule))

    def loss_fn(params, gates, target):
        probs, _ = forward(params, gates)
        return -jnp.sum(target * jnp.log(probs + eps))       # NLL; minimises forward KL

    def train_one(gates, params0, target):
        state0 = opt.init(params0)

        def step(carry, _):
            params, state = carry
            val, grads = jax.value_and_grad(loss_fn)(params, gates, target)
            updates, state = opt.update(grads, state, params)
            return (optax.apply_updates(params, updates), state), val

        (params, _), losses = jax.lax.scan(step, (params0, state0), None, length=n_epochs)
        return params, losses

    return jax.jit(jax.vmap(train_one, in_axes=(0, 0, None)))
