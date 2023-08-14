require "active_support/all"
require "matplotlib/pyplot"
require "torch-rb"

module Activation
  extend self

  def step_function_for_num(x)
    x.positive? ? 1 : 0
  end

  # x: (Tensor)
  def step_function(x)
    Torch.heaviside(x, Torch.tensor(0, dtype: x.min.dtype))
  end

  def plot_step_function
    plot(Torch.arange(-5.0, 5.0, 0.1), function: :step_function)
  end

  # x: (Tensor)
  def sigmoid(x)
    x.sigmoid
  end

  def plot_sigmoid
    plot(Torch.arange(-5.0, 5.0, 0.1), function: :sigmoid)
  end

  # x: (Tensor)
  def relu(x)
    x.relu
  end

  def plot_relu
    plot(Torch.arange(-5.0, 5.0, 0.1), function: :relu, ylim: [-1, 6])
  end

  def identity_function(x)
    x
  end

  def softmax(x)
    x.softmax dim: 0
  end

  private

    def plot(x, function: :sigmoid, ylim: [-0.1, 1.1])
      y = public_send(function, x)

      plt = Matplotlib::Pyplot
      plt.plot(x.to_a, y.to_a)
      plt.ylim(*ylim) if ylim.present?
      plt.show
    end
end
