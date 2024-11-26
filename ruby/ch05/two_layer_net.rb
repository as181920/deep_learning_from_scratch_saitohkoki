require "torch-rb"
require_relative "../common/global"
require_relative "../common/utility"
require_relative "../ch04/loss"
require_relative "../ch04/gradient"

class TwoLayerNet
  attr_reader :params, :layers, :last_layer

  def initialize(input_size: 784, hidden_size: 100, output_size: 10, weight_init_std: 0.01)
    initial_weights(input_size:, hidden_size:, output_size:, weight_init_std:)
    initial_layers
  end

  def predict(x)
    layers.each_value { |layer| x = layer.forward(x) }
    x
  end

  def loss(x, t)
    y = predict(x)
    last_layer.forward(y, t)
  end

  def accuracy(x, t)
    z = predict(x)
    y = Torch.argmax(z, dim: 1)
    t = Torch.argmax(t, dim: 1) if t.ndim != 1
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

  def gradient(x, t)
    # forward
    loss(x, t)

    # backward
    dout = 1
    last_layer.backward(dout)
    layers.each_value.reverse_each { |layer| dout = layer.backward(dout) }

    # set grads
    grads = {}
    grads["W1"] = layers["Affine1"].dw
    grads["b1"] = layers["Affine1"].db
    grads["W2"] = layers["Affine2"].dw
    grads["b2"] = layers["Affine2"].db
    grads
  end

  private

    def initial_weights(input_size:, hidden_size:, output_size:, weight_init_std:)
      @params = {}
      params["W1"] = Torch.randn(input_size, hidden_size, dtype: :float64).to(Global::DEVICE) * weight_init_std
      params["b1"] = Torch.zeros(hidden_size).to(Global::DEVICE)
      params["W2"] = Torch.randn(hidden_size, output_size, dtype: :float64).to(Global::DEVICE) * weight_init_std
      params["b2"] = Torch.zeros(output_size).to(Global::DEVICE)
    end

    def initial_layers
      @layers = {}
      layers["Affine1"] = AffineLayer.new(params["W1"], params["b1"])
      layers["Relu1"] = ReluLayer.new
      layers["Affine2"] = AffineLayer.new(params["W2"], params["b2"])

      @last_layer = SoftmaxWithLossLayer.new
    end
end
