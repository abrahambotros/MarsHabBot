# - https://thenewphalls.wordpress.com/2014/02/19/understanding-object-oriented-programming-in-julia-part-1/
# - http://stackoverflow.com/questions/26800811/julia-constructor-embedding-a-function-inside-a-type
type Visualizer
  ### variables
  pomdp::BotWorld
  CELL_SIZE::Int64
  fig
  ax
  placedTargets::Vector{Tuple{Int64, Int64}}

  #### functions
  init::Function
  #draw::Function
  #update::Function

  function Visualizer(pomdp::BotWorld; CELL_SIZE::Int64=1)
    self = new()
    self.pomdp = pomdp
    self.CELL_SIZE = CELL_SIZE

    self.init = function()

      # plot setup
      self.fig = figure()
      PyPlot.hold(true)
      PyPlot.ion()

      # general setup
      self.placedTargets = Vector{Tuple{Int64, Int64}}()

      # draw initial grid
      #state = getInitialBotState(self.pomdp)
      #self.draw()

    end

    # return self
    return self
  end
end


# - http://stackoverflow.com/questions/30351546/using-matplotlibs-patches-in-julia
# - http://matthiaseisen.com/pp/patterns/p0203/
# - http://matplotlib.org/api/patches_api.html
# - http://matplotlib.org/examples/shapes_and_collections/artist_reference.html
# - https://groups.google.com/forum/#!msg/julia-users/94P2OGaHVUA/oW2thTo4D1cJ
# - http://stackoverflow.com/questions/4098131/how-to-update-a-plot-in-matplotlib
# - http://stackoverflow.com/questions/12998430/remove-xticks-in-a-matplot-lib-plot
function visualizerDraw(self::Visualizer, prevState::BotState=getInitialBotState(self.pomdp), action::BotAction=POMDPs.create_action(self.pomdp), newState::BotState=getInitialBotState(self.pomdp))

  # clear
  PyPlot.clf()
  self.ax = self.fig[:add_subplot](1,1,1)

  # draw grid
  numCellsX = pomdp.WORLD_SIZE_X
  numCellsY = pomdp.WORLD_SIZE_Y
  PyPlot.xlim([-0.5,numCellsX*self.CELL_SIZE+0.5])
  PyPlot.ylim([-0.5,numCellsY*self.CELL_SIZE+0.5])
  numVerticalLines = numCellsX + 1
  numHorizontalLines = numCellsY + 1
  for i=1:numVerticalLines
    PyPlot.plot([self.CELL_SIZE*(i-1), self.CELL_SIZE*(i-1)], [0, self.CELL_SIZE*numCellsY], color="grey", alpha=0.2)
  end
  for i=1:numHorizontalLines
    PyPlot.plot([0, self.CELL_SIZE*numCellsX], [self.CELL_SIZE*(i-1), self.CELL_SIZE*(i-1)], color="grey", alpha=0.2)
  end

  # draw base
  #base = patch.RegularPolygon([pomdp.BASE_LOCATION[1]-0.5, pomdp.BASE_LOCATION[2]-0.5], 4, 0.4)
  base = patch.Rectangle([self.pomdp.BASE_LOCATION[1]-0.9, self.pomdp.BASE_LOCATION[2]-0.9], 0.8, 0.8, alpha=0.5, facecolor="blue")
  self.ax[:add_artist](base)


  # check if placed target this round
  if (prevState.isHoldingTarget == true) && (newState.isHoldingTarget == false) && (newState.numTargetsRemaining == prevState.numTargetsRemaining - 1)
    for targetSpecification in pomdp.TARGET_SPECIFICATIONS
      if (prevState.x == targetSpecification.x) && (prevState.y == targetSpecification.y)
        push!(self.placedTargets, (targetSpecification.x, targetSpecification.y))
      end
    end
  end

  # draw target positions
  for targetSpecification in pomdp.TARGET_SPECIFICATIONS
    # if not already placed
    if !( (targetSpecification.x, targetSpecification.y) in self.placedTargets )
      color = "green"
      fill = false
      linestyle = "dashed"
    # if already placed
    else
      color = "green"
      fill = true
      linestyle = "solid"
    end
    target = patch.RegularPolygon([targetSpecification.x-0.5, targetSpecification.y-0.5], 4, 0.3, alpha=0.8, facecolor=color, fill=fill, linestyle=linestyle)
    self.ax[:add_artist](target)
  end

  # draw obstacles
  for obstacleSpecification in pomdp.OBSTACLE_SPECIFICATIONS
    obstacle = patch.Circle([obstacleSpecification.x-0.5, obstacleSpecification.y-0.5], 0.3, alpha=0.8, facecolor="red", fill=true)
    self.ax[:add_artist](obstacle)
  end

  # draw robot
  if !isEqualStates(newState, getIntendedNextState(pomdp, prevState, action)) # something went unexpected
    color = "red"
  elseif newState.isHoldingTarget == true
    color = "green"
  else
    color = "black"
  end
  if newState.facingDirection == :up
    orientation = 0
  elseif newState.facingDirection == :left
    orientation = 90*pi/180
  elseif newState.facingDirection == :down
    orientation = 180*pi/180
  elseif newState.facingDirection == :right
    orientation = 270*pi/180
  end
  robot = patch.RegularPolygon([newState.x-0.5, newState.y-0.5], 3, 0.25, orientation=orientation, facecolor=color)
  self.ax[:add_artist](robot)

  # show
  #PyPlot.show()
  PyPlot.draw()

end
