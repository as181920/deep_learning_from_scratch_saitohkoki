require "torch-rb"
require_relative "../common/global"

class MulLayer
  attr_reader :x, :y

  def forward(x, y)
    @x = x
    @y = y

    x * y
  end

  def backward(dout)
    dx = dout * y
    dy = dout * x

    [dx, dy]
  end
end
