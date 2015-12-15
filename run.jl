using POMDPs
using POMDPDistributions
using DiscreteValueIteration
using MCTS

#using POMDPToolbox
#using SARSOP
#using QMDP

using PyPlot
using PyCall
@pyimport matplotlib.patches as patch


# TODO:
# - observations/pomdp
# - add tSinceLastRecalibration to BotState, implement as POMDP with noisy observations
# - add recalibration action, with some cost
# - might penalize for placing target when not high enough belief?

### conventions
# - world locations are (1,1) in bottom-left, (+x,+y) in top-right

### RUNTIME CONFIG
#WORLD_CONFIG = :small
#WORLD_CONFIG = :medium
WORLD_CONFIG = :large
assert( (WORLD_CONFIG == :small) || (WORLD_CONFIG == :medium) || (WORLD_CONFIG == :large) )
# certainty configs, useful for debugging, testing, comparisons
#CERTAINTY_CONFIG = :certain
#CERTAINTY_CONFIG = :uncertainMovement_uncertainTargetInteraction
#CERTAINTY_CONFIG = :uncertainMovement_certainTargetInteraction
CERTAINTY_CONFIG = :uncertainMovement_uncertainTargetInteraction_decreasedCertainty
#CERTAINTY_CONFIG = :uncertainMovement_certainTargetInteraction_decreasedCertainty
if CERTAINTY_CONFIG == :certain
  PROBABILITY_MOVEMENT_SUCCESS = 1.0 # probability of transitioning to the desired state when moving/turning in a specific direction
  PROBABILITY_TARGET_INTERACTION_SUCCESS = 1.0 # probability of successfully picking up/placing target object
elseif CERTAINTY_CONFIG == :uncertainMovement_uncertainTargetInteraction
  PROBABILITY_MOVEMENT_SUCCESS = 0.9
  PROBABILITY_TARGET_INTERACTION_SUCCESS = 0.9
elseif CERTAINTY_CONFIG == :uncertainMovement_certainTargetInteraction # useful for eliminating increased rewards for picking/placing target repeatedly due to world-induced failure
  PROBABILITY_MOVEMENT_SUCCESS = 0.9
  PROBABILITY_TARGET_INTERACTION_SUCCESS = 1.0
elseif CERTAINTY_CONFIG == :uncertainMovement_uncertainTargetInteraction_decreasedCertainty
  PROBABILITY_MOVEMENT_SUCCESS = 0.7
  PROBABILITY_TARGET_INTERACTION_SUCCESS = 0.7
elseif CERTAINTY_CONFIG == :uncertainMovement_certainTargetInteraction_decreasedCertainty
  PROBABILITY_MOVEMENT_SUCCESS = 0.7
  PROBABILITY_TARGET_INTERACTION_SUCCESS = 1.0
end
# solver config
RUN_DISCRETE_VALUE_ITERATION = true
RUN_SARSOP = false



### config constants
if WORLD_CONFIG == :small
  BASE_LOCATION = (5, 5) # location of base where targets originate
  WORLD_SIZE_X = 5 # number of blocks in world x dimension
  WORLD_SIZE_Y = 5 # number of blocks in world y dimension
  NUM_TOTAL_TARGETS = 4 # number of total targets the bot must place
  NUM_TOTAL_OBSTACLES = 3 # number of total obstacles the bot must avoid
elseif WORLD_CONFIG == :medium
  BASE_LOCATION = (1, 1) # location of base where targets originate
  WORLD_SIZE_X = 10 # number of blocks in world x dimension
  WORLD_SIZE_Y = 10 # number of blocks in world y dimension
  NUM_TOTAL_TARGETS = 6 # number of total targets the bot must place
  NUM_TOTAL_OBSTACLES = 6 # number of total obstacles the bot must avoid
elseif WORLD_CONFIG == :large
  BASE_LOCATION = (20, 10) # location of base where targets originate
  WORLD_SIZE_X = 20 # number of blocks in world x dimension
  WORLD_SIZE_Y = 15 # number of blocks in world y dimension
  NUM_TOTAL_TARGETS = 15 # number of total targets the bot must place
  NUM_TOTAL_OBSTACLES = 6 # number of total obstacles the bot must avoid
end



### functions for generating specific constants
# create dict with facingDirectionSymbol => Int64, for use in sub2ind
function createFacingDirectionSymbolsIndexDict(facingDirectionSymbols)
  indexDict = Dict{Symbol, Int64}()
  for (i, symbol) in enumerate(facingDirectionSymbols)
    indexDict[symbol] = i
  end
  return indexDict
end

### constants
ACTION_SYMBOLS = [:moveForward, :turnLeft, :turnRight, :pickUpTarget, :placeTarget] # all possible robot actions (as symbols)
NUM_ACTIONS = length(ACTION_SYMBOLS) # total number of possible actions
FACING_DIRECTION_SYMBOLS = [:up, :down, :left, :right] # all possible facing directions (as symbols)
FACING_DIRECTION_SYMBOLS_INDEX_DICT = createFacingDirectionSymbolsIndexDict(FACING_DIRECTION_SYMBOLS)
FACING_DIRECTION_ADDITION_VECTORS = Dict(:up => (0, +1), :down => (0, -1), :left => (-1, 0), :right => (+1, 0)) # dictionary for addition vectors based on facing directions; useful for computing new location if move forward when facing a particular direction
NUM_FACING_DIRECTIONS = length(FACING_DIRECTION_SYMBOLS) # total number of facing directions
NUM_MOVE_FORWARD_POSSIBLE_NEW_LOCATIONS = 4 # total number of new locations we can end up in when moving forward
NUM_TOTAL_STATES = (WORLD_SIZE_X+2) * (WORLD_SIZE_Y+2) * NUM_FACING_DIRECTIONS * (NUM_TOTAL_TARGETS+1) * 2 # *2 for each bitwise state variable, NUM_TOTAL_TARGETS+1 to allow 0 state, SIZE+2 to allow for invalid states on either side for each dimension for value iteration etc
OBSTACLE_ERROR_PENALTY_DEFAULT = -100 # penalty incurred if receive error at this default obstacle
REWARD_PICKUP_TARGET = 100.0 # reward for picking up a target from the base location
REWARD_PLACE_TARGET = 1000.0 # reward for successfully placing a target at its desired location
PENALTY_INVALID_STATE = -100 # penalty for entering invalid state (used for value iteration)
PENALTY_MOVEMENT = -1 # penalty for movement
PENALTY_TURN = -1 # penalty for turning/changing direction
PENALTY_OUT_OF_BOUNDS = -50 # penalty for moving out-of-bounds
PENALTY_INVALID_PICKUPTARGET_NOTBASE = -30 # penalty for trying to pick up a target when not at the base location
PENALTY_INVALID_PICKUPTARGET_ALREADYHOLDING = -100 # penalty for trying to pick up a target when already holding one
PENALTY_INVALID_PLACETARGET = -50 # penalty for trying to place a target when aren't actually holding one
PENALTY_WRONG_PLACETARGET = -100 # penalty for placing a target at the wrong location
PENALTY_STANDING_ON_PAST_TARGET_LOCATION = 0 # penalty if standing on past target location in current state, not necessarily meaning to get here. in practice, this is penalty if turning on location of target just placed
PENALTY_MOVING_TO_PAST_TARGET_LOCATION = -100 # penalty for walking on a target location where target has already been placed
DISCOUNT_FACTOR = 0.9 # discount factor



### BotState model
type BotState <: POMDPs.State
  x::Int64 # x position (ground-truth)
  y::Int64 # y position (ground-truth)
  facingDirection::Symbol # direction robot is facing
  numTargetsRemaining::Int64 # number of targets not yet completed/placed
  isHoldingTarget::Bool # true if currently holding a target, false otherwise
end

### Action type
type BotAction <: POMDPs.Action
  actionSymbol::Symbol
end

### Observation type
type BotObservation <: POMDPs.Observation
  xObs::Int64 # x position (observed, noisy)
  yObs::Int64 # y position (observed, noisy)
end

### Targets - target specification
type TargetSpecification
  x::Int64 # x position
  y::Int64 # y position
  orderIndex::Int64 # 1-indexed index of this target's place in overall order targets must be placed
end

### Obstacles - obstacle specification
type ObstacleSpecification
  x::Int64 # x position
  y::Int64 # y position
  #errorFraction::Float64 # fraction that this obstacle causes error penalty
  errorPenalty::Float64 # penalty incurred by error at this obstacle
end


### World model
# - See initial constants sections for comments on specific variables
type BotWorld <: POMDP
  WORLD_CONFIG::Symbol
  WORLD_SIZE_X::Int64
  WORLD_SIZE_Y::Int64
  BASE_LOCATION::Tuple{Int64, Int64}
  NUM_TOTAL_STATES::Int64
  NUM_MAX_STATE_NEIGHBORS::Int64
  STATES_INDEXES_MAP::Dict{BotState, Int64} # map from state to state index
  NUM_TOTAL_TARGETS::Int64
  ACTION_SYMBOLS::Array{Symbol}
  FACING_DIRECTION_SYMBOLS::Array{Symbol}
  FACING_DIRECTION_ADDITION_VECTORS::Dict{Symbol, Tuple{Int64, Int64}}
  FACING_DIRECTION_SYMBOLS_INDEX_DICT::Dict{Symbol, Int64}
  NUM_MOVE_FORWARD_POSSIBLE_NEW_LOCATIONS::Int64
  TARGET_SPECIFICATIONS::Vector{TargetSpecification} # list of targets for this pomdp world
  OBSTACLE_SPECIFICATIONS::Vector{ObstacleSpecification} # list of obstacles for this pomdp world
  REWARD_STATES_ACTIONS_VALUES::Vector{Tuple{BotState, BotAction, Float64}}
  PROBABILITY_MOVEMENT_SUCCESS::Float64
  PROBABILITY_TARGET_INTERACTION_SUCCESS::Float64
  REWARD_TARGET_PICKUP::Float64
  REWARD_TARGET_PLACEMENT::Float64
  PENALTY_INVALID_STATE::Float64
  PENALTY_MOVEMENT::Float64
  PENALTY_TURN::Float64
  PENALTY_OUT_OF_BOUNDS::Float64
  PENALTY_INVALID_PICKUPTARGET_NOTBASE::Float64
  PENALTY_INVALID_PICKUPTARGET_ALREADYHOLDING::Float64
  PENALTY_INVALID_PLACETARGET::Float64
  PENALTY_WRONG_PLACETARGET::Float64
  PENALTY_STANDING_ON_PAST_TARGET_LOCATION::Float64
  PENALTY_MOVING_TO_PAST_TARGET_LOCATION::Float64
  DISCOUNT_FACTOR::Float64
end


### StateSpace type
type StateSpace <: POMDPs.AbstractSpace
  states::Vector{BotState}
end
### StateSpace multiple-dispatch for POMDPs.jl
function POMDPs.states(pomdp::BotWorld)
  #states, _ = createStates(pomdp)
  #return StateSpace(states)
  return StateSpace(createStates(pomdp))
end
### StateSpace domain - return iterator over state space for POMDPs.jl
function POMDPs.domain(stateSpace::StateSpace)
  return stateSpace.states
end
### allow uniform sampling (in-place) of state space
function POMDPs.rand!(rng::AbstractRNG, state::BotState, stateSpace::StateSpace)
  #sampledState = stateSpace.states[rand(rng, 1:end)]
  copy!(state, stateSpace.states[rand(rng, 1:end)])
  return state
end
### actually create states
function createStates(pomdp::BotWorld)
  states, stateIndexesMap = createStates(pomdp.WORLD_SIZE_X, pomdp.WORLD_SIZE_Y, pomdp.FACING_DIRECTION_SYMBOLS, pomdp.NUM_TOTAL_TARGETS, pomdp.NUM_TOTAL_STATES)
  return states
  ## instantiate vector
  #states = BotState[]
  #stateIndexesMap = Dict{BotState, Int64}()
  #currentStateIndex = 1
  ## for each x location
  #for x=1:pomdp.WORLD_SIZE_X
  #  # for each y location
  #  for y=1:pomdp.WORLD_SIZE_Y
  #    # for each facing direction
  #    for facingDirection in pomdp.FACING_DIRECTION_SYMBOLS
  #      # for each numTargetsRemaining
  #      for numTargetsRemaining=0:pomdp.NUM_TOTAL_TARGETS
  #        # for each isHoldingTarget
  #        for isHoldingTarget in [true, false]
  #          # create state
  #          state = BotState(x, y, facingDirection, numTargetsRemaining, isHoldingTarget)
  #          # append state
  #          push!(states, state)
  #          #push!(states, BotState(x, y, facingDirection, numTargetsRemaining, isHoldingTarget))
  #          # get state index
  #          stateIndexesMap[state] = currentStateIndex
  #          # increment currentStateIndex
  #          currentStateIndex += 1
  #        end
  #      end
  #    end
  #  end
  #end
  ## assert
  #assert( length(states) == pomdp.NUM_TOTAL_STATES )
  #assert( length(stateIndexesMap) == pomdp.NUM_TOTAL_STATES )
  ## print
  #@printf("Total number of states: %d\n", pomdp.NUM_TOTAL_STATES)
  ## return
  #return states, stateIndexesMap
end
# NOTE: also creates invalid-location states around the edge
function createStates(worldSizeX, worldSizeY, facingDirectionSymbols, numTotalTargets, numTotalStates)
  # instantiate vector
  states = BotState[]
  stateIndexesMap = Dict{BotState, Int64}()
  currentStateIndex = 1
  # for each x location
  for x=0:worldSizeX+1
    # for each y location
    for y=0:worldSizeY+1
      # for each facing direction
      for facingDirection in facingDirectionSymbols
        # for each numTargetsRemaining
        for numTargetsRemaining=0:numTotalTargets
          # for each isHoldingTarget
          for isHoldingTarget in [true, false]
            # create state
            state = BotState(x, y, facingDirection, numTargetsRemaining, isHoldingTarget)
            # append state
            push!(states, state)
            #push!(states, BotState(x, y, facingDirection, numTargetsRemaining, isHoldingTarget))
            # get state index
            stateIndexesMap[state] = currentStateIndex
            # increment currentStateIndex
            currentStateIndex += 1
          end
        end
      end
    end
  end
  # assert
  assert( length(states) == numTotalStates )
  assert( length(stateIndexesMap) == numTotalStates )
  # print
  @printf("Total number of states: %d\n", numTotalStates)
  # return
  return states, stateIndexesMap
end
### helper function for MCTS
Base.hash(state::BotState, h::UInt64 = zero(UInt64)) = hash(state.x, hash(state.y, hash(state.facingDirection, hash(state.numTargetsRemaining, hash(state.isHoldingTarget, h)))))
### parse a passed-in state into a tuple of its components
function parseBotState(state::BotState)
  return state.x, state.y, state.facingDirection, state.numTargetsRemaining, state.isHoldingTarget
end
### get initial state
function getInitialBotState(pomdp::BotWorld)
  return BotState(pomdp.BASE_LOCATION[1], pomdp.BASE_LOCATION[2], :up, pomdp.NUM_TOTAL_TARGETS, false)
end
### get default state
function getDefaultBotState(pomdp::BotWorld)
  return BotState(1, 1, :ANY_DIRECTION, pomdp.NUM_TOTAL_TARGETS, false)
end
### get invalid state, which can't be equal to any valid states
function getInvalidBotState(pomdp::BotWorld)
  return BotState(-1, -1, :ANY_DIRECTION, -1, false)
end
### make copy of inputState, place into outputState in-place
function Base.copy!(outputState::BotState, inputState::BotState)
  outputState.x = inputState.x
  outputState.y = inputState.y
  outputState.facingDirection = inputState.facingDirection
  outputState.numTargetsRemaining = inputState.numTargetsRemaining
  outputState.isHoldingTarget = inputState.isHoldingTarget
  return outputState
end
### get max number of state neighbors at any given state
# using actions [:moveForward, :turnLeft, :turnRight, :pickUpTarget, :placeTarget]
function getMaxNumStateNeighbors()
  # - can move forward - get forward, same, or forward-left, or forward-right (4)
  # - can face new direction by turning, or face same direction (3)
  # - can have new number of targets remaining, or same (2)
  # - can be holding or not holding target (2)
  # - since we can only pick one action at a time, we take the max of the above options
  return 4
end
### determine if two states are equal, optionally ignoring specific params
function isEqualStates(s1::BotState, s2::BotState; ignoreDirection=false)
  if (ignoreDirection == false) && (s1.facingDirection != s2.facingDirection)
    return false
  end
  return (s1.x == s2.x) && (s1.y == s2.y) && (s1.numTargetsRemaining == s2.numTargetsRemaining) && (s1.isHoldingTarget == s2.isHoldingTarget)
  #if (s1.x != s2.x)
  #  return false
  #elseif (s1.y != s2.y)
  #  return false
  #elseif (s1.numTargetsRemaining != s2.numTargetsRemaining)
  #  return false
  #elseif (s1.facingDirection != s2.facingDirection) && (ignoreDirection == false)
  #  return false
  #elseif (s1.isHoldingTarget != s2.isHoldingTarget)
  #  return false
  #else
  #  return true
  #end
end
### determine if state and TargetSpecification are equal in location
function isEqualLocations(state::BotState, targetSpecification::TargetSpecification)
  return (state.x == targetSpecification.x) && (state.y == targetSpecification.y)
end
### determine if state and ObstacleSpecification are equal in location
function isEqualLocations(state::BotState, obstacleSpecification::ObstacleSpecification)
  return (state.x == obstacleSpecification.x) && (state.y == obstacleSpecification.y)
end
### initializer function
function POMDPs.create_state(pomdp::BotWorld)
  return getInitialBotState(pomdp)
end
### helper function for MCTS
Base.isequal(s1::BotState, s2::BotState) = isEqualStates(s1, s2)
### determine if passed-in state is at base location state
function isAtBaseLocation(pomdp::BotWorld, state::BotState)
  return ( (state.x == pomdp.BASE_LOCATION[1]) && (state.y == pomdp.BASE_LOCATION[2]) )
end
### determine if passed-in state is a valid state
function isValidState(pomdp::BotWorld, state::BotState)

  # if out-of-bounds state
  if isOutOfBounds(pomdp, state)
    return false
  end

  # if no targets remaining but holding a target
  if (state.numTargetsRemaining == 0) && (state.isHoldingTarget == true)
    return false
  end

  # otherwise, valid
  return true
end
### determine if passed-in state is a terminal state
function isTerminalState(state::BotState)
  return (state.numTargetsRemaining == 0) #&& (isHoldingTarget == false)# && (isPlacingTarget == false)
end
function MCTS.isterminal(pomdp::BotWorld, state::BotState)
  return isTerminalState(state)
end
### determine if passed-in state has a location that is out-of-bounds
function isOutOfBounds(pomdp::BotWorld, state::BotState)
  return ( (state.x <= 0) || (state.x > pomdp.WORLD_SIZE_X) || (state.y <= 0) || (state.y > pomdp.WORLD_SIZE_Y) )
end
### get intended/expected next state if successfully execute action from prev state
### - NOTE: does not check for validity of this action in this state, just assumes it can happen as commanded (important: will not check if at target location for picking up/placing target; do this elsewhere)
function getIntendedNextState(pomdp::BotWorld, currentState::BotState, action::BotAction)

  # copy currentState to nextState
  nextState = getDefaultBotState(pomdp)
  copy!(nextState, currentState)

  # parse currentState
  currentX, currentY, currentFacingDirection, currentNumTargetsRemaining, currentIsHoldingTarget = parseBotState(currentState)
  # get action symbol from action
  actionSymbol = action.actionSymbol

  # if moveForward
  if actionSymbol == :moveForward
    # get new location if move forward one step in direction we are currently facing
    dx, dy = pomdp.FACING_DIRECTION_ADDITION_VECTORS[currentFacingDirection]
    nextState.x += dx
    nextState.y += dy

  # elseif turnLeft
  elseif actionSymbol == :turnLeft
    if currentFacingDirection == :up
      nextState.facingDirection = :left
    elseif currentFacingDirection == :left
      nextState.facingDirection = :down
    elseif currentFacingDirection == :down
      nextState.facingDirection = :right
    elseif currentFacingDirection == :right
      nextState.facingDirection = :up
    else
      @printf("getIntendedNextState: error with actionSymbol=%s, currentFacingDirection=%s\n", actionSymbol, currentFacingDirection)
      error("getIntendedNextState: invalid parameters")
    end
  # elseif turnRight
  elseif actionSymbol == :turnRight
    if currentFacingDirection == :up
      nextState.facingDirection = :right
    elseif currentFacingDirection == :right
      nextState.facingDirection = :down
    elseif currentFacingDirection == :down
      nextState.facingDirection = :left
    elseif currentFacingDirection == :left
      nextState.facingDirection = :up
    else
      @printf("getIntendedNextState: error with actionSymbol=%s, currentFacingDirection=%s\n", actionSymbol, currentFacingDirection)
      error("getIntendedNextState: invalid parameters")
    end

  # elseif pickUpTarget - NOTE: not checked for validity
  elseif actionSymbol == :pickUpTarget
    nextState.isHoldingTarget = true

  # elseif :placeTarget - NOTE: not checked for validity
  elseif actionSymbol == :placeTarget
    nextState.isHoldingTarget = false
    nextState.numTargetsRemaining -= 1

  # otherwise, invalid actionSymbol
  else
    @printf("getIntendedNextState: error, invalid actionSymbol=%s\n", actionSymbol)
    error("getIntendedNextState: invalid parameters")
  end

  # return
  return nextState
end
### Returns possible new locations if moving forward from a current state
### - First is current state (to work with transitions function format)
### - Second is intended state (moving forward one step)
### - Third is moving forward one, then left one (relative to currentFacingDirection)
### - Fourth is moving forward one, then right one (relative to currentFacingDirection)
function getMoveForwardPossibleNewLocations(pomdp::BotWorld, state::BotState)

  # instantiate
  newLocations = Vector{BotState}()
  for i=1:pomdp.NUM_MOVE_FORWARD_POSSIBLE_NEW_LOCATIONS
    push!(newLocations, getInvalidBotState(pomdp))
  end

  # copy current state into first slot
  copy!(newLocations[1], state)

  # get intended forward state
  intendedNextState = getIntendedNextState(pomdp, state, BotAction(:moveForward))
  # copy into second slot
  copy!(newLocations[2], intendedNextState)

  # get forward-left and forward-right states
  currentFacingDirection = state.facingDirection
  copy!(newLocations[3], intendedNextState)
  copy!(newLocations[4], intendedNextState)
  if currentFacingDirection == :up
    newLocations[3].x -= 1
    newLocations[4].x += 1
  elseif currentFacingDirection == :left
    newLocations[3].y -= 1
    newLocations[4].y += 1
  elseif currentFacingDirection == :down
    newLocations[3].x += 1
    newLocations[4].x -= 1
  elseif currentFacingDirection == :right
    newLocations[3].y += 1
    newLocations[4].y -= 1
  else
    error("getMoveForwardPossibleNewLocations: invalid currentFacingDirection")
  end

  # return
  return newLocations
end
### return state that is nearest to outOfBoundsState, but in bounds
function moveToNearestInBoundsLocation(pomdp::BotWorld, outOfBoundsState::BotState)
  inBoundsState = getInvalidBotState(pomdp)
  copy!(inBoundsState, outOfBoundsState)
  if inBoundsState.x < 1
    inBoundsState.x = 1
  elseif inBoundsState.x > pomdp.WORLD_SIZE_X
    inBoundsState.x = pomdp.WORLD_SIZE_X
  end
  if inBoundsState.y < 1
    inBoundsState.y = 1
  elseif inBoundsState.y > pomdp.WORLD_SIZE_Y
    inBoundsState.y = pomdp.WORLD_SIZE_Y
  end
  assert( !isOutOfBounds(pomdp, inBoundsState) )
  return inBoundsState
end


### ActionSpace type
type ActionSpace <: AbstractSpace
  actions::Vector{BotAction}
end
### ActionSpace multiple-dispatch for POMDPs.jl
function POMDPs.actions(pomdp::BotWorld)
  return ActionSpace(createActions(pomdp))
end
function POMDPs.actions(pomdp::BotWorld, state::BotState, actionSpace::ActionSpace=POMDPs.actions(pomdp))
  return actionSpace
end
### ActionSpace domain - return iterator over action space for POMDPs.jl
function POMDPs.domain(actionSpace::ActionSpace)
  return actionSpace.actions
end
### allow uniform sampling (in-place) of action space
function POMDPs.rand!(rng::AbstractRNG, action::BotAction, actionSpace::ActionSpace)
  #sampledAction = actionSpace.actions[rand(rng, 1:end)]
  copy!(action, actionSpace.actions[rand(rng, 1:end)])
  return action
end
### actually create actions
function createActions(pomdp::BotWorld)
  # instantiate vector
  actions = BotAction[]
  # create BotActions from symbols
  for actionSymbol in pomdp.ACTION_SYMBOLS
    push!(actions, BotAction(actionSymbol))
  end
  # print
  @printf("Total number of actions: %d\n", length(actions))
  # return
  return actions
end
### initializer function
function POMDPs.create_action(pomdp::BotWorld)
  return BotAction(:moveForward)
end
### determine if two actions are equivalent
function isEqualActions(a1::BotAction, a2::BotAction)
  return (a1.actionSymbol == a2.actionSymbol)
end
### make copy of inputAction, place into outputAction in-place
function Base.copy!(outputAction::BotAction, inputAction::BotAction)
  outputAction.actionSymbol = inputAction.actionSymbol
  return outputAction
end



### Transition distribution
type BotWorldDistribution <: POMDPs.AbstractDistribution
  neighbors::Vector{BotState} # the states s' in the distribution
  probabilities::Vector{Float64} # array of probabilities for each corresponding state s'
  categorical::POMDPDistributions.Categorical # comes from POMDPDistributions.jl and is used for sampling
end
### To reduce memory allocation, the POMDPs.jl interface defines some initialization functions that return initial
### types to be filled later. This function returns the distribution type filled with some values. We don't care
### what the distribution container has in it, because it will be modified at each call to the transition function.
function POMDPs.create_transition_distribution(pomdp::BotWorld)
  #neighbors = [BotState(1, 1, :ANY_DIRECTION, NUM_TOTAL_TARGETS, false, false) for i=1:NUM_MAX_STATE_NEIGHBORS]
  neighbors = [getDefaultBotState(pomdp) for i=1:pomdp.NUM_MAX_STATE_NEIGHBORS]
  probabilities = zeros(pomdp.NUM_MAX_STATE_NEIGHBORS) + 1.0/pomdp.NUM_MAX_STATE_NEIGHBORS
  categorical = POMDPDistributions.Categorical(pomdp.NUM_MAX_STATE_NEIGHBORS)
  return BotWorldDistribution(neighbors, probabilities, categorical)
end
### return iterator over discrete states in distribution (neighbors array in our distribution)
function POMDPs.domain(distribution::BotWorldDistribution)
  return distribution.neighbors
end
### implement PDF/PMF function from POMDPS.jl - for discrete distribution, pdf function returns probability of
### drawing the state s from the distribution d
function POMDPs.pdf(distribution::BotWorldDistribution, targetState::BotState)
  for (i, candidateState) in enumerate(distribution.neighbors)
    #if candidateState == targetState
    if isEqualStates(candidateState, targetState)
      return distribution.probabilities[i]
    end
  end
  return 0.0
end
### implement sampling function that can draw samples from our distribution
function POMDPs.rand!(rng::AbstractRNG, state::BotState, distribution::BotWorldDistribution)
  set_prob!(distribution.categorical, distribution.probabilities) # fill the Categorical distribbution with our state probabilities
  #sampledNeighbor = distribution.neighbors
  copy!(state, distribution.neighbors[rand(rng, distribution.categorical)]) # sample a neighbor state according to the distribution
  return state
end
### implement transition function
function POMDPs.transition(pomdp::BotWorld, currentState::BotState, action::BotAction, distribution::BotWorldDistribution=create_transition_distribution(pomdp))

  ### SETUP

  # get reference to neighbors from given distribution variable
  neighbors = distribution.neighbors
  probabilities = distribution.probabilities

  # fill with all zeros, copy current state into first neighbor for convenience
  fill!(probabilities, 0.0)
  copy!(neighbors[1], currentState)
  # copy empty states into other neighbors to make sure we don't have duplicates of currentState/etc, as this causes problems with pdf
  for i=2:length(neighbors)
    copy!(neighbors[i], getInvalidBotState(pomdp))
  end
  #assert(isEqualStates(neighbors[1], currentState))

  ### PARSE CURRENTSTATE AND ACTION FOR CONVENIENCE
  currentX, currentY, currentFacingDirection, currentNumTargetsRemaining, currentIsHoldingTarget = parseBotState(currentState)
  actionSymbol = action.actionSymbol
  #currentX = currentState.x
  #currentY = currentState.y
  #currentFacingDirection = currentState.facingDirection
  #currentNumTargetsRemaining = currentState.numTargetsRemaining
  #currentIsHoldingTarget = currentState.isHoldingTarget
  #currentIsPlacingTarget = currentState.isPlacingTarget

  ### GET INTENDEDNEXTSTATE
  intendedNextState = getIntendedNextState(pomdp, currentState, action)
  ## place into second slot
  #copy!(neighbors[2], intendedNextState)


  ### HANDLE BEING IN OUT-OF-BOUNDS STATE - return to nearest in-bounds state immediately (incur penalty in rewards function already)
  if isOutOfBounds(pomdp, currentState)
    copy!(neighbors[1], moveToNearestInBoundsLocation(pomdp, currentState))
    probabilities[1] = 1.0
    return distribution
  end

  ### HANDLE BEING IN INVALIDSTATE FROM START - stay in invalid state forever
  if !isValidState(pomdp, currentState)
    probabilities[1] = 1.0
    return distribution
  end


  ### HANDLE DETERMINISTIC TRANSITIONS INDEPENDENT OF ACTION

  ## if currently placing target, then transition deterministically to next state where we are no longer placing, are no longer holding, and have one fewer targets to worry about
  #if (currentIsHoldingTarget == true) && (currentIsPlacingTarget == true)
  #  neighbors[1].isHoldingTarget = false
  #  neighbors[1].numTargetsRemaining -= 1
  #  probabilities[1] = 1.0
  #  return distributions
  #end
    

  ### handle terminal state case first
  if isTerminalState(currentState)
    # only transition to same state deterministically
    #fill!(probabilities, 0.0)
    probabilities[1] = 1.0
    #copy!(neighbors[1], currentState)
    return distribution # when sample distributions in future, will only get the state in neighbors[1] - our terminal state
  end


  ### set neighbors/probabilities based on actions
#ACTION_SYMBOLS = [:moveForward, :turnLeft, :turnRight, :pickUpTarget, :placeTarget] # all possible robot actions (as symbols)



  # if moveForward
  if actionSymbol == :moveForward

    # get appropriate probability p
    pNextState = pomdp.PROBABILITY_MOVEMENT_SUCCESS

    ## if intended next state will take us out of bounds
    #if isOutOfBounds(pomdp, intendedNextState)

    # get all moveForwardPossibleNewLocations
    # - first/same state already in neighbors[1]
    # - intended state is in newLocations[2]
    # - front-left and front-right in newLocations[3 and 4]
    moveForwardPossibleNewLocations = getMoveForwardPossibleNewLocations(pomdp, currentState)

    # copy intendedNextState into second neighbor
    assert( isEqualStates(intendedNextState, moveForwardPossibleNewLocations[2]) )
    copy!(neighbors[2], moveForwardPossibleNewLocations[2])

    # copy 3rd and 4th into 3rd and 4th neighbors
    copy!(neighbors[3], moveForwardPossibleNewLocations[3])
    copy!(neighbors[4], moveForwardPossibleNewLocations[4])

    # 2nd probability is intended; divide the rest equally
    pOther = (1-pNextState)/3
    probabilities[1] = pOther
    probabilities[2] = pNextState
    probabilities[3] = pOther
    probabilities[4] = pOther

    ## if nextState is still in bounds
    ##if !isOutOfBounds(pomdp, intendedNextState)
    #  # copy intendedNextState into second neighbor
    #  copy!(neighbors[2], intendedNextState)
    #  # stay in same state wth prob=(1-p), move to new state with prob=(p)
    #  probabilities[1] = 1-pNextState
    #  probabilities[2] = pNextState

    ## otherwise, stay in same state deterministically
    #else
    #  probabilities[1] = 1.0
    #end

    ## with probability PROBABILITY_MOVEMENT_SUCCESS
    #if rand() <= pomdp.PROBABILITY_MOVEMENT_SUCCESS

    #  # move forward one step
    #  #nextState = getIntendedNextState(pomdp, currentState, action)
    #  #(newX, newY) = getForwardLocationGivenFacingDirection(pomdp, currentStateX, currentStateY, currentFacingDirection)
    #  # move only if nextState still in bounds
    #  if !isOutOfBounds(pomdp, intendedNextState)
    #    copy!(neighbors[1], intendedNextState)
    #    #neighbors[1].x = nextState.x #newX
    #    #neighbors[1].y = nextState.y #newY
    #  end
    #end

    ## all probability goes to being in state one step ahead (or not, if movement failed), with everything else fixed
    #probabilities[1] = 1.0



  # if turnLeft/turnRight
  elseif actionSymbol == :turnLeft || actionSymbol == :turnRight

    # get appropriate probability p
    pNextState = pomdp.PROBABILITY_MOVEMENT_SUCCESS

    # copy intendedNextState into second neighbor
    copy!(neighbors[2], intendedNextState)
    # stay in same state with prob=(1-p), move to new state with prob=(p)
    probabilities[1] = 1-pNextState
    probabilities[2] = pNextState


    ## with probability PROBABILITY_MOVEMENT_SUCCESS
    #if rand() <= pomdp.PROBABILITY_MOVEMENT_SUCCESS

    #  # face new intended direction successfully
    #  #nextState = getIntendedNextState(pomdp, currentState, action)
    #  #copy!(neighbors[1], nextState)
    #  copy!(neighbors[1], intendedNextState)#getIntendedNextState(podmp, currentState, action))
    #  #newFacingDirection = getIntendedNewFacingDirection(currentFacingDirection, action)
    #  #neighbors[1].facingDirection = newFacingDirection

    #end

    ## all probability goes to being in this new state (or same state if movement failed)
    #probabilities[1] = 1.0



  # if pickUpTarget
  elseif actionSymbol == :pickUpTarget

    # get appropriate probability p
    pNextState = pomdp.PROBABILITY_TARGET_INTERACTION_SUCCESS

    # if at base location, and not already holding target
    if (isAtBaseLocation(pomdp, currentState)) && (currentIsHoldingTarget == false) && (currentNumTargetsRemaining > 0)

      # copy intendedNextState into second neighbor
      copy!(neighbors[2], intendedNextState)
      # stay in same state with prob=(1-p), move to new state with prob=(p)
      probabilities[1] = 1-pNextState
      probabilities[2] = pNextState

    # otherwise, invalid action, and just deterministically stay in same state
    else
      probabilities[1] = 1.0
    end


    ## with probability PROBABILITY_TARGET_INTERACTION_SUCCESS
    #if rand() <= pomdp.PROBABILITY_TARGET_INTERACTION_SUCCESS

    #  # if at base location, and not already holding target
    #  if (isAtBaseLocation(pomdp, currentState)) && (currentIsHoldingTarget == false) && (currentNumTargetsRemaining > 0)
    #  #if ((currentX, currentY) == pomdp.BASE_LOCATION) && (currentIsHoldingTarget == false)

    #    # pick up target
    #    #copy!(neighbors[1], getIntendedNextState(pomdp, currentState, action))
    #    copy!(neighbors[1], intendedNextState)
    #    #neighbors[1].isHoldingTarget = true

    #  end
    #end

    ## all probability goes to being in this new state (or same state if interaction failed)
    #probabilities[1] = 1.0



  # if placeTarget
  elseif actionSymbol == :placeTarget

    # get appropriate probability p
    pNextState = pomdp.PROBABILITY_TARGET_INTERACTION_SUCCESS

    # get past, current, future targets
    pastTargetSpecifications, currentTargetSpecification, futureTargetSpecifications = getTargetSpecificationsPartitionedByPlaced(pomdp, currentState)

    # if already holding target and at least one target remaining for us to hold, and if correct location
    if (currentIsHoldingTarget == true) && (currentNumTargetsRemaining > 0) && (isEqualLocations(currentState, currentTargetSpecification))

      # copy intendedNextState into second neighbor
      copy!(neighbors[2], intendedNextState)
      # stay in same state with prob=(1-p), move to new state with prob=(p)
      probabilities[1] = 1-pNextState
      probabilities[2] = pNextState

    # otherwise, invalid action, and just deterministically stay in same state
    else
      probabilities[1] = 1.0
    end
    ## with probability PROBABILITY_TARGET_INTERACTION_SUCCESS
    #if rand() <= pomdp.PROBABILITY_TARGET_INTERACTION_SUCCESS

    #  # if already holding target and at least one target remaining for us to hold
    #  if (currentIsHoldingTarget == true) && (currentNumTargetsRemaining > 0) #&& (currentIsPlacingTarget == false)

    #    # place target
    #    copy!(neighbors[1], intendedNextState)
    #    #neighbors[1].isHoldingTarget = false
    #    #neighbors[1].numTargetsRemaining -= 1

    #  end
    #end

    ## all probability goes to being in this new state (or same state if interaction failed)
    #probabilities[1] = 1.0



  # otherwise, invalid action
  else
    # print and error
    @printf("POMDPs.transition: got invalid action. currentState=%s, action=%s\n", currentState, action)
    error("POMDPs.transition: got invalid action.")
  end
  

  # return full distribution, containing neighbors and probabilities
  return distribution
end











### Targets and obstacles - create targets and obstacles list
function createTargetAndObstacleSpecifications(worldConfig, worldSizeX, worldSizeY, numTotalTargets, numTotalObstacles)

  # print
  @printf("Building targets and obstacles for worldConfig:%s. Size:(%d, %d). Targets:%d. Obstacles:%d\n", worldConfig, worldSizeX, worldSizeY, numTotalTargets, numTotalObstacles)

  # instantiate
  targetSpecifications = Vector{TargetSpecification}() # list of targets for this pomdp world
  obstacleSpecifications = Vector{ObstacleSpecification}() # list of obstacles for this pomdp world

  # if :small world
  if WORLD_CONFIG == :small

    # config
    #BASE_LOCATION = (5, 5) # location of base where targets originate
    #WORLD_SIZE_X = 5 # number of blocks in world x dimension
    #WORLD_SIZE_Y = 5 # number of blocks in world y dimension
    #NUM_TOTAL_TARGETS = 4 # number of total targets the bot must place
    #NUM_TOTAL_OBSTACLES = 3 # number of total obstacles the bot must avoid

    # targets
    push!(targetSpecifications, TargetSpecification(3, 3, 1))
    push!(targetSpecifications, TargetSpecification(3, 4, 2))
    push!(targetSpecifications, TargetSpecification(4, 3, 3))
    push!(targetSpecifications, TargetSpecification(4, 4, 4))
    # obstacles
    push!(obstacleSpecifications, ObstacleSpecification(2, 4, OBSTACLE_ERROR_PENALTY_DEFAULT))
    push!(obstacleSpecifications, ObstacleSpecification(2, 5, OBSTACLE_ERROR_PENALTY_DEFAULT))
    push!(obstacleSpecifications, ObstacleSpecification(3, 5, OBSTACLE_ERROR_PENALTY_DEFAULT))


  # if :medium world
  elseif WORLD_CONFIG == :medium

    # config
    #BASE_LOCATION = (1, 1) # location of base where targets originate
    #WORLD_SIZE_X = 10 # number of blocks in world x dimension
    #WORLD_SIZE_Y = 10 # number of blocks in world y dimension
    #NUM_TOTAL_TARGETS = 6 # number of total targets the bot must place
    #NUM_TOTAL_OBSTACLES = 6 # number of total obstacles the bot must avoid

    # targets
    push!(targetSpecifications, TargetSpecification(5, 4, 1))
    push!(targetSpecifications, TargetSpecification(6, 4, 2))
    push!(targetSpecifications, TargetSpecification(7, 5, 3))
    push!(targetSpecifications, TargetSpecification(6, 6, 4))
    push!(targetSpecifications, TargetSpecification(5, 6, 5))
    push!(targetSpecifications, TargetSpecification(4, 5, 6))

    # obstacles
    push!(obstacleSpecifications, ObstacleSpecification(1, 4, OBSTACLE_ERROR_PENALTY_DEFAULT))
    push!(obstacleSpecifications, ObstacleSpecification(2, 5, OBSTACLE_ERROR_PENALTY_DEFAULT))
    push!(obstacleSpecifications, ObstacleSpecification(4, 1, OBSTACLE_ERROR_PENALTY_DEFAULT))
    push!(obstacleSpecifications, ObstacleSpecification(7, 2, OBSTACLE_ERROR_PENALTY_DEFAULT))
    push!(obstacleSpecifications, ObstacleSpecification(8, 7, OBSTACLE_ERROR_PENALTY_DEFAULT))
    push!(obstacleSpecifications, ObstacleSpecification(9, 7, OBSTACLE_ERROR_PENALTY_DEFAULT))

  elseif WORLD_CONFIG == :large

    # config
    #BASE_LOCATION = (20, 10) # location of base where targets originate
    #WORLD_SIZE_X = 20 # number of blocks in world x dimension
    #WORLD_SIZE_Y = 15 # number of blocks in world y dimension
    #NUM_TOTAL_TARGETS = 10 # number of total targets the bot must place
    #NUM_TOTAL_OBSTACLES = 10 # number of total obstacles the bot must avoid

    # targets
    push!(targetSpecifications, TargetSpecification(6, 7, 1))
    push!(targetSpecifications, TargetSpecification(7, 6, 2))
    push!(targetSpecifications, TargetSpecification(7, 8, 3))
    push!(targetSpecifications, TargetSpecification(8, 5, 4))
    push!(targetSpecifications, TargetSpecification(8, 9, 5))
    push!(targetSpecifications, TargetSpecification(9, 6, 6))
    push!(targetSpecifications, TargetSpecification(9, 8, 7))
    push!(targetSpecifications, TargetSpecification(10, 7, 8))
    push!(targetSpecifications, TargetSpecification(11, 6, 9))
    push!(targetSpecifications, TargetSpecification(11, 8, 10))
    push!(targetSpecifications, TargetSpecification(12, 5, 11))
    push!(targetSpecifications, TargetSpecification(12, 9, 12))
    push!(targetSpecifications, TargetSpecification(13, 6, 13))
    push!(targetSpecifications, TargetSpecification(13, 8, 14))
    push!(targetSpecifications, TargetSpecification(14, 7, 15))

    # obstacles
    push!(obstacleSpecifications, ObstacleSpecification(5, 5, OBSTACLE_ERROR_PENALTY_DEFAULT))
    push!(obstacleSpecifications, ObstacleSpecification(5, 9, OBSTACLE_ERROR_PENALTY_DEFAULT))
    push!(obstacleSpecifications, ObstacleSpecification(10, 5, OBSTACLE_ERROR_PENALTY_DEFAULT))
    push!(obstacleSpecifications, ObstacleSpecification(10, 9, OBSTACLE_ERROR_PENALTY_DEFAULT))
    push!(obstacleSpecifications, ObstacleSpecification(14, 5, OBSTACLE_ERROR_PENALTY_DEFAULT))
    push!(obstacleSpecifications, ObstacleSpecification(14, 9, OBSTACLE_ERROR_PENALTY_DEFAULT))



  # otherwise
  else
    # throw error - undefined scenario
    error("Undefined world for creating targets and obstacles.")
  end

  # assert
  assert( length(targetSpecifications) == numTotalTargets ) # must have appropriate total number of targets/specifications
  for (i, targetSpecification) in enumerate(targetSpecifications) # must be in order, or at least have one target specification for each order index
    assert( targetSpecification.orderIndex == i )
  end
  assert( length(obstacleSpecifications) == numTotalObstacles ) # must have appropriate total number of obstacles/specifications

  # print
  #@printf("Total number of targets: %d, total number of obstacles: %d\n", numTotalTargets, numTotalObstacles)

  # return
  return targetSpecifications, obstacleSpecifications
end
  ## create targets and obstacles for the (1,1) world with 1 target
  #if worldSizeX == 1 && worldSizeY == 2 && numTotalTargets == 1
  #  # target 1: (1, 2)
  #  push!(targetSpecifications, TargetSpecification(1, 2, 1))

  ## create targets and obstacles for the (2,2) world with 1 target
  #elseif worldSizeX == 2 && worldSizeY == 2 && numTotalTargets == 1
  #  # target 1: (1, 2)
  #  push!(targetSpecifications, TargetSpecification(1, 2, 1))

  ## create targets and obstacles for the (2,2) world with 2 targets
  #elseif worldSizeX == 2 && worldSizeY == 2 && numTotalTargets == 2
  #  # target 1: (1, 2)
  #  push!(targetSpecifications, TargetSpecification(1, 2, 1))
  #  # target 2: (2, 2)
  #  push!(targetSpecifications, TargetSpecification(2, 2, 2))

  ## create targets and obstacles for the (3,3) world with 2 targets
  #elseif worldSizeX == 3 && worldSizeY == 3 && numTotalTargets == 2
  #  # target 1: (1, 3)
  #  push!(targetSpecifications, TargetSpecification(1, 3, 1))
  #  # target 2: (3, 3)
  #  push!(targetSpecifications, TargetSpecification(3, 3, 2))

  ## create targets and obstacles for the (5,5) world with 2 targets
  #elseif worldSizeX == 5 && worldSizeY == 5 && numTotalTargets == 4 && numTotalObstacles == 3
  #  # targets
  #  push!(targetSpecifications, TargetSpecification(4, 4, 1))
  #  push!(targetSpecifications, TargetSpecification(4, 3, 2))
  #  push!(targetSpecifications, TargetSpecification(3, 4, 3))
  #  push!(targetSpecifications, TargetSpecification(3, 3, 4))
  #  # obstacles
  #  push!(obstacleSpecifications, ObstacleSpecification(1, 3, OBSTACLE_ERROR_PENALTY_DEFAULT))
  #  push!(obstacleSpecifications, ObstacleSpecification(1, 4, OBSTACLE_ERROR_PENALTY_DEFAULT))
  #  push!(obstacleSpecifications, ObstacleSpecification(2, 4, OBSTACLE_ERROR_PENALTY_DEFAULT))
  # 

  ## create targets and obstacles for the (5,5) world with 2 targets
  #elseif worldSizeX == 7 && worldSizeY == 7 && numTotalTargets == 2
  #  # target 1: (1, 7)
  #  push!(targetSpecifications, TargetSpecification(1, 7, 1))
  #  # target 2: (7, 7)
  #  push!(targetSpecifications, TargetSpecification(7, 7, 2))

  ## create targets and obstacles for the (10,10) world with 2 targets
  #elseif worldSizeX == 10 && worldSizeY == 10 && numTotalTargets == 2
  #  # target 1: (1, 10)
  #  push!(targetSpecifications, TargetSpecification(2, 10, 1))
  #  # target 10: (10, 10)
  #  push!(targetSpecifications, TargetSpecification(10, 10, 2))

  ## create targets and obstacles for the (10,10) world with 6 targets and 4 obstacles
  #elseif worldSizeX == 10 && worldSizeY == 10 && numTotalTargets == 6 && numTotalObstacles == 6
  #  # target 1: (5, 4)
  #  push!(targetSpecifications, TargetSpecification(5, 4, 1))
  #  # target 2: (6, 4)
  #  push!(targetSpecifications, TargetSpecification(6, 4, 2))
  #  # target 3: (7, 5)
  #  push!(targetSpecifications, TargetSpecification(7, 5, 3))
  #  # target 4: (6, 6)
  #  push!(targetSpecifications, TargetSpecification(6, 6, 4))
  #  # target 5: (5, 6)
  #  push!(targetSpecifications, TargetSpecification(5, 6, 5))
  #  # target 6: (4, 5)
  #  push!(targetSpecifications, TargetSpecification(4, 5, 6))

  #  # obstacles
  #  push!(obstacleSpecifications, ObstacleSpecification(1, 4, OBSTACLE_ERROR_PENALTY_DEFAULT))
  #  push!(obstacleSpecifications, ObstacleSpecification(2, 5, OBSTACLE_ERROR_PENALTY_DEFAULT))
  #  push!(obstacleSpecifications, ObstacleSpecification(4, 1, OBSTACLE_ERROR_PENALTY_DEFAULT))
  #  push!(obstacleSpecifications, ObstacleSpecification(7, 2, OBSTACLE_ERROR_PENALTY_DEFAULT))
  #  push!(obstacleSpecifications, ObstacleSpecification(8, 7, OBSTACLE_ERROR_PENALTY_DEFAULT))
  #  push!(obstacleSpecifications, ObstacleSpecification(9, 7, OBSTACLE_ERROR_PENALTY_DEFAULT))


### partition targetSpecifications as past/alreadyPlaced, present/currentTarget, and future/futureTargets based on state
function getTargetSpecificationsPartitionedByPlaced(pomdp::BotWorld, state::BotState)

  # init
  past = Vector{TargetSpecification}()
  current = Vector{TargetSpecification}()
  future = Vector{TargetSpecification}()

  # get number of targets already placed according to state
  numAlreadyPlacedTargets = pomdp.NUM_TOTAL_TARGETS - state.numTargetsRemaining

  # iterate
  for targetSpecification in pomdp.TARGET_SPECIFICATIONS
    # if already placed
    if targetSpecification.orderIndex <= numAlreadyPlacedTargets
      # add to past list
      push!(past, targetSpecification)
    # else if current target
    elseif targetSpecification.orderIndex == (numAlreadyPlacedTargets + 1)
      # add to current list
      push!(current, targetSpecification)
    # otherwise, future target
    else
      # add to future list
      push!(future, targetSpecification)
    end
  end

  # check
  assert( length(past) + length(current) + length(future) == pomdp.NUM_TOTAL_TARGETS )
  assert( length(current) == 1 )

  # return
  return past, current[1], future
end
#### get targetSpecification for state's current target
#function getCurrentTargetSpecification(pomdp::BotWorld, state::BotState)
#  # get number of targets already placed according to state
#  numAlreadyPlacedTargets = pomdp.NUM_TOTAL_TARGETS - state.numTargetsRemaining
#
#  # get the next target (that hasn't been placed)
#  for targetSpecification in pomdp.TARGET_SPECIFICATIONS
#    if targetSpecification.orderIndex == (numAlreadyPlacedTargets + 1)
#      return targetSpecification
#    end
#  end
#
#  # otherwise (if already placed all targets, for example), return empty
#  return nothing
#end
#### get vector of targetSpecifications that have already been placed according to state
#function getAlreadyPlacedTargetSpecifications(pomdp::BotWorld, state::BotState)
#
#  # init
#  alreadyPlacedTargetSpecifications = Vector{TargetSpecification}()
#
#  # get number of targets already placed according to state
#  numAlreadyPlacedTargets = pomdp.NUM_TOTAL_TARGETS - state.numTargetsRemaining
#
#  # get first numAlreadyPlacedTargets targetSpecifications
#  for targetSpecification in pomdp.TARGET_SPECIFICATIONS
#    if targetSpecification.orderIndex <= numAlreadyPlacedTargets
#      push!(alreadyPlacedTargetSpecifications, targetSpecification)
#    end
#  end
#
#  # return
#  return alreadyPlacedTargetSpecifications
#end
#### determine if passed-in state is at current target location state (doesn't check if holding target; do this elsewhere)
#function isAtCurrentTargetLocation(pomdp::BotWorld, state::BotState)
#
#  # get current state's current target
#  currentTargetSpecification = getCurrentTargetSpecification(pomdp, state)
#
#  # return true if at currentTargetSpecification's location
#  return (state.x == currentTargetSpecification.x) && (state.y == currentTargetSpecification.y)
#end

  #### - note that rewardStates we loop through must have holdingTarget=true meaning they are placeTarget location, not pickUpTarget location
  ## if at base location, then not at target location
  #if isAtBaseLocation(pomdp, state)
  #  return false
  #end
  ## for each rewardStatesActionsValues
  #for (rewardState, rewardAction, rewardValue) in rewardStatesActionsValues
  #  # check if currentState has rewardState location and numTargetsRemaining
  #  if (state.x == rewardState.x) && (state.y == rewardState.y) && (state.numTargetsRemaining == rewardState.numTargetsRemaining) && (rewardState.isHoldingTarget == true)
  #    return true
  #  end
  #end
  ## otherwise, return false
  #return false
#end





### Rewards - rewardStatesActionsValues
function createRewards(worldSizeX, worldSizeY, numTotalTargets, baseLocation, targetSpecifications, rewardPickUpTarget, rewardPlaceTarget)

  # instantiate
  rewardStatesActionsValues = Vector{Tuple{BotState, BotAction, Float64}}() # when in BotState and take BotAction, receive reward
  baseLocationX = baseLocation[1]
  baseLocationY = baseLocation[2]
  actionPickUpTarget = BotAction(:pickUpTarget)
  actionPlaceTarget = BotAction(:placeTarget)

  # for each targetSpecification
  for (i, targetSpecification) in enumerate(targetSpecifications)

    # compute number of targets that must be remaining BEFORE PLACING THIS TARGET AT ITS TARGET LOCATION based on targetSpecification's order index
    numTargetsRemaining = numTotalTargets - targetSpecification.orderIndex + 1

    # create rewardStateActionValue tuple for picking up this target from base
    push!(rewardStatesActionsValues, (BotState(baseLocationX, baseLocationY, :ANY_DIRECTION, numTargetsRemaining, false), actionPickUpTarget, rewardPickUpTarget))
    # create rewardStateActionValue tuple for placing at target location
    push!(rewardStatesActionsValues, (BotState(targetSpecification.x, targetSpecification.y, :ANY_DIRECTION, numTargetsRemaining, true), actionPlaceTarget, rewardPlaceTarget))

  end

  ## create rewardStateActionValue for ending state
  #push!(rewardStatesActionsValues, (BotState(baseLocationX, baseLocationY, :ANY_DIRECTION, 0, false)

  # assert
  assert( length(rewardStatesActionsValues) == numTotalTargets*2 ) # *2 since count for each pickup and placement

  # print
  @printf("Total number of reward states: %d\n", length(rewardStatesActionsValues))

  # return
  return rewardStatesActionsValues

end



### Rewards - R(s,a) model
function POMDPs.reward(pomdp::BotWorld, state::BotState, action::BotAction)

  ## get reward model
  #rewardStatesActionsValues = pomdp.REWARD_STATES_ACTIONS_VALUES
  # instantiate reward
  r = 0.0
  # get expected next state
  intendedNextState = getIntendedNextState(pomdp, state, action)
  # get actionSymbol
  actionSymbol = action.actionSymbol

  # if invalid state, then return penalty
  if !isValidState(pomdp, state)
    #@printf("Got invalid state: %s\n", state)
    return pomdp.PENALTY_INVALID_STATE
  end

  # if terminal state already, then no more reward
  if isTerminalState(state)
    return 0.0
  end

  # for each reward state-action-reward tuple
  for (rewardState, rewardAction, rewardValue) in pomdp.REWARD_STATES_ACTIONS_VALUES
    # if currently in rewardState and taking rewardAction
    if isEqualStates(state, rewardState, ignoreDirection=true) && isEqualActions(action, rewardAction)
      ## TODO: REMOVE
      #@printf("In reward state: state=%s, rewardState=%s\n", state, rewardState)
      # rewarded with rewardValue
      r += rewardValue
    end
  end



  # but penalty for walking on target location if already placed target there
  # - note: if always penalize regardless of placing there yet or not, then can get stuck in middle spinning around!
  pastTargetSpecifications, currentTargetSpecification, futureTargetSpecification = getTargetSpecificationsPartitionedByPlaced(pomdp, state)
  for pastTargetSpecification in pastTargetSpecifications
    # penalty if current state is on target location that has already been placed
    if isEqualLocations(state, pastTargetSpecification)
      r += pomdp.PENALTY_STANDING_ON_PAST_TARGET_LOCATION
    end
    # penalty if trying to move to target location that has already been placed. turning is ok, since we likely just placed this target, and might need to avoid moving forward into an enclosed area/trap
    if isEqualLocations(intendedNextState, pastTargetSpecification) && (actionSymbol != :turnLeft && actionSymbol != :turnRight)
      r += pomdp.PENALTY_MOVING_TO_PAST_TARGET_LOCATION
    end
  end
  #for targetSpecification in pomdp.TARGET_SPECIFICATIONS
  #  if isEqualLocations(state, targetSpecification)
  #    r += pomdp.PENALTY_MOVING_TO_TARGET_LOCATION
  #  end
  #  if isEqualLocations(intendedNextState, targetSpecification)
  #    r += pomdp.PENALTY_MOVING_TO_TARGET_LOCATION
  #  end
  #end

  # TODO: also penalty for ending on target too...
  # penalty for being on or trying to move onto obstacle location
  for obstacleSpecification in pomdp.OBSTACLE_SPECIFICATIONS
    # TODO: comment this out again?
    if isEqualLocations(state, obstacleSpecification)
      r += obstacleSpecification.errorPenalty
    end
    if isEqualLocations(intendedNextState, obstacleSpecification)
      r += obstacleSpecification.errorPenalty
    end
  end

  # penalty for movement
  if actionSymbol == :moveForward
    r += pomdp.PENALTY_MOVEMENT
  end

  # penalty for turn
  if (actionSymbol == :turnLeft) || (actionSymbol == :turnRight)
    r += pomdp.PENALTY_TURN
  end

  # penalty for trying to go out of bounds
  if isOutOfBounds(pomdp, intendedNextState)
    r += pomdp.PENALTY_OUT_OF_BOUNDS
  end

  # penalty for trying to pick up a target when not at the base location, or if already have one
  if actionSymbol == :pickUpTarget
    # if not at base location
    if !isAtBaseLocation(pomdp, state)
      r += pomdp.PENALTY_INVALID_PICKUPTARGET_NOTBASE
    end
    # if picking up when already have one
    if state.isHoldingTarget == true
      r += pomdp.PENALTY_INVALID_PICKUPTARGET_ALREADYHOLDING
    end
  end

  # penalty for placing a target at an incorrect location, or if don't have a target already
  if actionSymbol == :placeTarget
    # get partitioned targets based on current state
    pastTargetSpecifications, currentTargetSpecification, futureTargetSpecification = getTargetSpecificationsPartitionedByPlaced(pomdp, state)
    # if at incorrect location for current target
    if !isEqualLocations(state, currentTargetSpecification)
    #if !isAtTargetLocation(pomdp, state)
      r += pomdp.PENALTY_WRONG_PLACETARGET
    end
    # if placing but don't actually have a target on us
    if state.isHoldingTarget == false
      r += pomdp.PENALTY_INVALID_PLACETARGET
    end
  end

  # return
  return r
end


### observation initializer function
function POMDPs.create_observation(pomdp::BotWorld)
  baseLocation = pomdp.BASE_LOCATION
  return BotObservation(baseLocation[1], baseLocation[2])
end
### ObservationSpace
type BotObservationSpace <: POMDPs.AbstractSpace
  observations::Vector{BotObservation}
end
### functions returning observation space
function POMDPs.observations(pomdp::BotWorld)
  return BotObservationSpace([POMDPs.create_observation(pomdp)])
end
function POMDPs.observations(pomdp::BotWorld, state::BotState, observationSpace::BotObservationSpace)
  return observationSpace
end
### function returning iterator over observation space
function POMDPs.domain(observationSpace::BotObservationSpace)
  return observationSpace.observations
end
### observation distribution type
type BotObservationDistribution <: POMDPs.AbstractDistribution
  probabilities::Vector{Float64}
  observations::Vector{BotObservation}
end
### observation distribution initializer
function POMDPs.create_observation_distribution(pomdp::BotWorld)
  return BotObservationDistribution([1.0], [POMDPs.create_observation(pomdp)])
end
### observation pdf
function POMDPs.pdf(distribution::BotObservationDistribution, observation::BotObservation)
  return distribution.probabilities[1]
end
### sample from observation distribution
function POMDPs.rand!(rng::AbstractRNG, observation::BotObservation, distribution::BotObservationDistribution)
  categorical = Categorical(distribution.probabilities)
  #observation.xObs = distribution.observations[sampledObservationIndex]sampledObservation.xObs
  sampledObservationIndex = rand(rng, categorical)
  sampledObservation = distribution.observations[sampledObservationIndex]
  observation.xObs = sampledObservation.xObs
  observation.yObs = sampledObservation.yObs
  return observation
end
### observation model
function POMDPs.observation(pomdp::BotWorld, state::BotState, action::BotAction, distribution::BotObservationDistribution=POMDPs.create_observation_distribution(pomdp))
  probabilities = distribution.probabilities
  observations = distribution.observations
  observations[1] = BotObservation(state.x, state.y)
  probabilities[1] = 1.0
  return distribution
end

### BELIEFS
### initialization function
function POMDPs.create_belief(pomdp::BotWorld)
  @printf("length(POMDPs.domain(POMDPS.states(pomdp))) == %d\n", length(POMDPs.domain(POMDPs.states(pomdp))))
  return POMDPToolbox.DiscreteBelief(length(POMDPs.domain(POMDPs.states(pomdp))))
end
# initial belief is same as create_belief
function POMDPs.initial_belief(pomdp::BotWorld)
  @printf("length(POMDPs.domain(POMDPS.states(pomdp))) == %d\n", length(POMDPs.domain(POMDPs.states(pomdp))))
  return POMDPToolbox.DiscreteBelief(length(POMDPs.domain(POMDPs.states(pomdp))))
  #return POMDPToolbox.DiscreteBelief(1)
end





### miscellaneous functions
function POMDPs.n_states(pomdp::BotWorld)
  return pomdp.NUM_TOTAL_STATES
end
function POMDPs.n_actions(pomdp::BotWorld)
  return length(pomdp.ACTION_SYMBOLS)
end
function POMDPs.n_observations(pomdp::BotWorld)
  return 1
end
function POMDPs.discount(pomdp::BotWorld)
  return pomdp.DISCOUNT_FACTOR
end
# have to make modificatoins so that no 0 indexes can appear
function POMDPs.index(pomdp::BotWorld, state::BotState)
  #println(state)
  #return sub2ind( (pomdp.WORLD_SIZE_X, pomdp.WORLD_SIZE_Y, length(pomdp.FACING_DIRECTION_SYMBOLS), pomdp.NUM_TOTAL_TARGETS+1, 2), state.x, state.y, pomdp.FACING_DIRECTION_SYMBOLS_INDEX_DICT[state.facingDirection], state.numTargetsRemaining+1, Int(state.isHoldingTarget+1) )
  return pomdp.STATES_INDEXES_MAP[state]
end




### run

# print
println("Begin")
timeStart = time()

## create states
#states = createStates()

# create state space - has array .states
#stateSpace = createStateSpace()

# do any dynamic setup needed
NUM_MAX_STATE_NEIGHBORS = getMaxNumStateNeighbors()

# create state indexes map based on world params
_, statesIndexesMap = createStates(WORLD_SIZE_X, WORLD_SIZE_Y, FACING_DIRECTION_SYMBOLS, NUM_TOTAL_TARGETS, NUM_TOTAL_STATES)

# create targets and obstacles based on world params
targetSpecifications, obstacleSpecifications = createTargetAndObstacleSpecifications(WORLD_CONFIG, WORLD_SIZE_X, WORLD_SIZE_Y, NUM_TOTAL_TARGETS, NUM_TOTAL_OBSTACLES)
# create rewards based on world params
rewardStatesActionsValues = createRewards(WORLD_SIZE_X, WORLD_SIZE_Y, NUM_TOTAL_TARGETS, BASE_LOCATION, targetSpecifications, REWARD_PICKUP_TARGET, REWARD_PLACE_TARGET)
#rewardStates, rewardValues = createRewards()

# additional asserts
assert( (WORLD_SIZE_X > 0) && (WORLD_SIZE_Y > 0) )
assert( NUM_MAX_STATE_NEIGHBORS > 0 )
assert( (length(FACING_DIRECTION_SYMBOLS) == length(FACING_DIRECTION_ADDITION_VECTORS)) && (length(FACING_DIRECTION_SYMBOLS) == length(FACING_DIRECTION_SYMBOLS_INDEX_DICT)) )
assert( NUM_MOVE_FORWARD_POSSIBLE_NEW_LOCATIONS == 4 )
assert( (0 <= PROBABILITY_MOVEMENT_SUCCESS) && (PROBABILITY_MOVEMENT_SUCCESS <= 1) )
assert( (0 <= PROBABILITY_TARGET_INTERACTION_SUCCESS) && (PROBABILITY_TARGET_INTERACTION_SUCCESS <= 1) )
assert( REWARD_PICKUP_TARGET > 0 )
assert( REWARD_PLACE_TARGET > 0 )
assert( PENALTY_INVALID_STATE < 0 )
assert( PENALTY_MOVEMENT < 0 )
assert( PENALTY_TURN < 0 )
assert( PENALTY_OUT_OF_BOUNDS < 0 )
assert( PENALTY_INVALID_PICKUPTARGET_NOTBASE < 0 )
assert( PENALTY_INVALID_PICKUPTARGET_ALREADYHOLDING < 0 )
assert( PENALTY_INVALID_PLACETARGET < 0 )
assert( PENALTY_WRONG_PLACETARGET < 0 )
assert( PENALTY_STANDING_ON_PAST_TARGET_LOCATION <= 0 )
assert( PENALTY_MOVING_TO_PAST_TARGET_LOCATION < 0 )
assert( (0 <= DISCOUNT_FACTOR) && (DISCOUNT_FACTOR <= 1) )

# instantiate world
pomdp = BotWorld(
          WORLD_CONFIG,
          WORLD_SIZE_X,
          WORLD_SIZE_Y,
          BASE_LOCATION,
          NUM_TOTAL_STATES,
          NUM_MAX_STATE_NEIGHBORS,
          statesIndexesMap,
          NUM_TOTAL_TARGETS,
          ACTION_SYMBOLS,
          FACING_DIRECTION_SYMBOLS,
          FACING_DIRECTION_ADDITION_VECTORS,
          FACING_DIRECTION_SYMBOLS_INDEX_DICT,
          NUM_MOVE_FORWARD_POSSIBLE_NEW_LOCATIONS,
          targetSpecifications,
          obstacleSpecifications,
          rewardStatesActionsValues,
          #rewardStates,
          #rewardValues,
          PROBABILITY_MOVEMENT_SUCCESS,
          PROBABILITY_TARGET_INTERACTION_SUCCESS,
          REWARD_PICKUP_TARGET,
          REWARD_PLACE_TARGET,
          PENALTY_INVALID_STATE,
          PENALTY_MOVEMENT,
          PENALTY_TURN,
          PENALTY_OUT_OF_BOUNDS,
          PENALTY_INVALID_PICKUPTARGET_NOTBASE,
          PENALTY_INVALID_PICKUPTARGET_ALREADYHOLDING,
          PENALTY_INVALID_PLACETARGET,
          PENALTY_WRONG_PLACETARGET,
          PENALTY_STANDING_ON_PAST_TARGET_LOCATION,
          PENALTY_MOVING_TO_PAST_TARGET_LOCATION,
          DISCOUNT_FACTOR)


### DISCRETE VALUE ITERATION

if RUN_DISCRETE_VALUE_ITERATION
  # initialize the policy by passing in problem
  policy = DiscreteValueIteration.ValueIterationPolicy(pomdp)
  # initialize solver
  solver = DiscreteValueIteration.ValueIterationSolver(max_iterations=200, belres=1e-3)
  # solve for an optimal policy
  DiscreteValueIteration.solve(solver, pomdp, policy, verbose=true)
end


#### MCTS
#solver = MCTS.MCTSSolver(n_iterations=1000, depth=100, exploration_constant=50.0)
#policy = MCTS.MCTSPolicy(solver, pomdp)


### SARSOP
if RUN_SARSOP

  # initialize policy; argument is name of policy file
  policy = POMDPPolicy("bot.policy")
  # create .pomdpx file - format which the SARSOP backend reads in
  pomdpxFile = POMDPFile(pomdp, "bot.pomdpx")
  # initialize solver
  solver = SARSOPSolver()
  # solve
  solve(solver, pomdpxFile, policy)
  # print
  println("Done solving sarsop")
  println(size(SARSOP.alphas(policy)))

  # sarsop init
  observation = create_observation(pomdp)
  belief = initial_belief(pomdp)
  updater = DiscreteUpdater(pomdp)
  action_map = POMDPs.domain(POMDPs.actions(pomdp))

end

#### QMDP
## initializer solver
#solver = QMDP.QMDPSolver(max_iterations=100, tolerance=1e-3)
## initialize policy
#policy = QMDP.create_policy(solver, pomdp)
## solve
#QMDP.solve(solver, pomdp, policy, verbose=true)


# simulation setup
println("Beginning simulation")
rng = MersenneTwister(round(Int, rand()*1000))
currentState = getInitialBotState(pomdp)
nextState = getInitialBotState(pomdp)
iterations = 1
rewardTotal = 0.0

# visualization setup
include("Visualizer.jl")
visualizer = Visualizer(pomdp)
visualizer.init()
visualizerDraw(visualizer)
firstIteration = true


# loop simulation
while currentState.numTargetsRemaining > 0 #&& iterations < 100

  # get optimal action from currentState according to policy
  if RUN_DISCRETE_VALUE_ITERATION
    action = DiscreteValueIteration.action(policy, currentState)
  elseif RUN_SARSOP
    # get action from SARSOP policy
    ai = SARSOP.action(policy, belief)
    action = action_map[ai]
  end

  # compute reward
  reward = POMDPs.reward(pomdp, currentState, action)
  rewardTotal += reward

  # transition to next state
  transitionDistribution = POMDPs.transition(pomdp, currentState, action)
  POMDPs.rand!(rng, nextState, transitionDistribution)

  # sample new observation
  if RUN_SARSOP
    observationDistribution = POMDPs.observation(pomdp, currentState, action)
    rand!(rng, observation, observationDistribution)
    # update belief
    belief = update(updater, belief, action, observation)
  end

  # print
  #@printf("s=%s, a=%s, s'=%s, r=%0.2f\n", currentState, action, nextState, reward)
  if RUN_DISCRETE_VALUE_ITERATION
    @printf("s=%s, a=%s, s'=%s, r=%0.2f\n", currentState, action, nextState, reward)
  elseif RUN_SARSOP
    @printf("s=%s, a=%s, s'=%s, r=%0.2f, o=%s\n", currentState, action, nextState, reward, observation)
  end

  # draw states
  visualizerDraw(visualizer, currentState, action, nextState)
  PyPlot.pause(0.03)

  # if first iteration, then stop here
  if firstIteration == true
    firstIteration = false
    userInput = readline(STDIN)
  end

  # copy next state into current state
  copy!(currentState, nextState)

  # increment iterations
  iterations += 1

end
@printf("RewardTotal=%0.2f\n", rewardTotal)

# finish
timeFinish = time() - timeStart
@printf("Done. Took %0.2f seconds\n", timeFinish)
userInput = readline(STDIN)
