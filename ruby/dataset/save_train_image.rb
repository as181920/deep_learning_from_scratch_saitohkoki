require_relative "mnist"

index = [59999, ARGV[0].to_i].min
Mnist.save_train_image(index)
