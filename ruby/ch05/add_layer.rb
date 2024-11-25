require "torch-rb"
require_relative "../common/global"

class AddLayer
  # attr_reader :x, :y

  def forward(x, y)
    x + y
  end

  def backward(dout)
    dx = dout * 1
    dy = dout * 1

    [dx, dy]
  end
end
