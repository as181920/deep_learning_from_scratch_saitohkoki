require "torch-rb"

module Utility
  module_function

  def softmax_from_scratch(x)
    x -= x.max(0)[0]
    x.exp / x.exp.sum
  end

  def softmax(x, dim: 0)
    x.softmax(dim:)
  end
end
