require "torch-rb"

module Perceptron
  module_function

  def and_gate(x1, x2)
    w1, w2, theta = 0.5, 0.5, 0.7 # rubocop:disable Style/ParallelAssignment
    ((x1 * w1) + (x2 * w2)).then { |sum| sum <= theta ? 0 : 1 }
  end

  def and_gate_with_bias(x1, x2)
    x = Torch.tensor([x1, x2])
    w = Torch.tensor([0.5, 0.5])
    b = -0.7
    ((x * w).sum + b) <= 0 ? 0 : 1
  end
end
