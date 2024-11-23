require "torch-rb"
require_relative "../common/utility"
require_relative "loss"

class SimpleNet
  attr_reader :weights

  # use shape 2x3 for simple net weights
  def initialize
    @weights = Torch.randn(2, 3, dtype: :float64)
  end

  def predict(x)
    Torch::Linalg.multi_dot([x, weights])
  end

  def loss(x, t)
    z = predict(x)
    y = Utility.softmax(z)
    Loss.cross_entropy_error(y, t)
  end
end
