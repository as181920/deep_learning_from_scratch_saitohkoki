require "torch-rb"
require_relative "../common/utility"
require_relative "loss"
require_relative "gradient"

class TwoLayerNet
  attr_reader :params

  def initialize(input_size: 784, hidden_size: 100, output_size: 10, weight_init_std: 0.01)
    @params = {}
    params["W1"] = Torch.randn(input_size, hidden_size, dtype: :float64) * weight_init_std
    params["b1"] = Torch.zeros(hidden_size)
    params["W2"] = Torch.randn(hidden_size, output_size, dtype: :float64) * weight_init_std
    params["b2"] = Torch.zeros(output_size)
  end

  def predict(x)
    a1 = Torch::Linalg.multi_dot([x, params["W1"]]) + params["b1"]
    z1 = Torch.sigmoid(a1)
    a2 = Torch::Linalg.multi_dot([z1, params["W2"]]) + params["b2"]
    Utility.softmax(a2)
  end

  def loss(x, t)
    y = predict(x)
    Loss.cross_entropy_error(y, t)
  end

  def accuracy(x, t)
    z = predict(x)
    y = Torch.argmax(z, dim: 1)
    t = Torch.argmax(t, dim: 1)
    Torch.eq(y, t).sum / Torch.tensor(x.shape[0], dtype: :float64)
  end

  def numerical_gradient(x, t)
    loss_f = ->(_w) { loss(x, t) }
    grads = {}
    grads["W1"] = Gradient.numerical_gradient(loss_f, params["W1"])
    grads["b1"] = Gradient.numerical_gradient(loss_f, params["b1"])
    grads["W2"] = Gradient.numerical_gradient(loss_f, params["W2"])
    grads["b2"] = Gradient.numerical_gradient(loss_f, params["b2"])
    grads
  end
end
