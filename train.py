import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from data import create_data_loader, evaluate

def create_train_state(rng, model, learning_rate):
    params = model.init({'params': rng, 'dropout': rng}, jnp.ones((1, 32, 32, 3)))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

def train_step(state, batch, dropout_rng):
    x, y = batch
    x = jnp.array(x)
    y = jnp.array(y)
    
    def loss_fn(params):
        rngs = {'dropout': dropout_rng}
        logits = state.apply_fn({'params': params}, x, rngs=rngs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train(model, train_data, val_data, batch_size, epochs, print_interval):
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model, learning_rate=0.001)
    
    for epoch in range(epochs):
        train_loader = create_data_loader(train_data, batch_size)
        val_loader = create_data_loader(val_data, batch_size)
        
        total_loss = 0
        batch_count = 0
        dropout_rng = jax.random.fold_in(rng, epoch)
        
        for batch in train_loader:
            batch_dropout_rng = jax.random.fold_in(dropout_rng, batch_count)
            state, loss = train_step(state, batch, batch_dropout_rng)
            total_loss += loss
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        if (epoch + 1) % print_interval == 0:
            accuracy = evaluate(state, val_loader, deterministic=True)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return state
