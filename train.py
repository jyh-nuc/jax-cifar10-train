import jax
import jax.numpy as jnp
import optax
from flax import nnx

def train(model, train_loader, val_loader, epochs=10, print_interval=1):
    optimizer = optax.adam(learning_rate=1e-3)
    params = model.init(jax.random.PRNGKey(0))['params']
    opt_state = optimizer.init(params)
    
    for epoch in range(epochs):
        train_loss = 0.0
        for x, y in train_loader:
            def loss_fn(p):
                model.update(p)
                logits = model(x)
                return jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, 10)))
            
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            train_loss += loss
        
        avg_train_loss = train_loss / len(train_loader)
        
        val_loss = 0.0
        for x, y in val_loader:
            model.update(params)
            logits = model(x)
            val_loss += jnp.mean(optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, 10)))
        avg_val_loss = val_loss / len(val_loader)
        
        if (epoch + 1) % print_interval == 0:
            print(f"Epoch {epoch+1:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    model.update(params)
    return model
