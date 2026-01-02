lume = require 'lume/lume'
lurker = require 'lurker/lurker'
game_engine = require 'game_engine'
neat = require 'neat'

local function calculate_fitness_for_wiggle(world)
  if (world.platform.body:getX() < 0) or (world.platform.body:getX() > world.width) then
    world.failed = true
  else
    world.genome.fitness = world.genome.fitness + math.cos(world.platform.body:getAngle() - math.rad(45))
  end
end

local max_movement_speed = 400

local function new_world(genome)
  local genome = genome or neat.create_genome({ input_count = 3, output_count = 1 })
  local world = game_engine.new_world()
  world.genome = genome
  world.canvas = love.graphics.newCanvas(world.width, world.height)
  return world
end

local function destroy(world)
  game_engine.destroy(world)
end

local function update(world, dt)
  game_engine.update(world, dt)

  calculate_fitness_for_wiggle(world)

  local dx, dy = world.platform.body:getLinearVelocity()
  local angle = world.stick.body:getAngle()
  local lv = world.stick.body:getAngularVelocity()

  local sx = 0
  local results = neat.evaluate(world.genome, { dx, angle, lv })
  for _, v in ipairs(results) do
    sx = v
  end
  world.platform.body:setLinearVelocity(sx * max_movement_speed, 0)
end

local function evaluate(worlds)
  local genomes = {}
  local failed_count = 0
  for _, v in ipairs(worlds) do
    table.insert(genomes, v.genome)
    if v.failed then
      failed_count = failed_count + 1
    end
  end

  if failed_count == #worlds then
    new_genomes = neat.evolve_population(genomes)
    for i, v in ipairs(worlds) do
      destroy(v)
      worlds[i] = new_world(new_genomes[i])
    end
  end

  return worlds
end


local worlds = {}
for i = 1, 10 do
  table.insert(worlds, new_world())
end

function love.update(dt)
  lurker.update()
  evaluate(worlds)

  for _, v in ipairs(worlds) do
    update(v, dt)
  end
end

function love.keypressed(key)
end

function love.draw()
  love.graphics.setColor(1, 1, 1, 1)
  love.graphics.print('Hello world')
  if #worlds == 0 then
    return
  end

  local s = love.graphics.getWidth() / (#worlds * worlds[1].width)

  for i, v in ipairs(worlds) do
    v.canvas:renderTo(function()
      love.graphics.clear()
      game_engine.draw(v)
    end)
    love.graphics.draw(v.canvas, (i - 1) * v.width * s, 0, 0, s, s)
  end
end
