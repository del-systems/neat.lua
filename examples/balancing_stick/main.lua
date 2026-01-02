lume = require 'lume/lume'
lurker = require 'lurker/lurker'
game_engine = require 'game_engine'
neat = require 'neat'

local function calculate_fitness_for_wiggle(world)
  if world.genome.failed then
    return
  end

  if (world.platform.body:getX() < 0) or (world.platform.body:getX() > world.width) then
    world.genome.failed = true
  else
    world.genome.fitness = world.genome.fitness + math.abs(math.sin(world.stick.body:getAngle()))
  end
end

local max_movement_speed = 400
local use_existing_model = true

local function new_world(genome)
  if not genome then
    genome = neat.create_genome({ input_count = 3, output_count = 1 })
    if use_existing_model then
      lume.each(genome.connections, function (c) c.enabled = false end)
      genome.nodes[4].activation = neat.get_default_activation_by_name("tanh")
      table.insert(genome.nodes, { id = 5, type = "hidden", activation = neat.get_default_activation_by_name("relu") })
      table.insert(genome.connections, { in_node = 1, out_node = 4, weight = 0.75, enabled = true, innovation = 2 })
      table.insert(genome.connections, { in_node = 2, out_node = 4, weight = 2.08, enabled = true, innovation = 3 })
      table.insert(genome.connections, { in_node = 3, out_node = 5, weight = 0.76, enabled = true, innovation = 4 })
      table.insert(genome.connections, { in_node = 5, out_node = 4, weight = 0.26, enabled = true, innovation = 5 })
    end
  end
  local world = game_engine.new_world()
  world.genome = genome
  genome.fitness = 0
  world.canvas = love.graphics.newCanvas(world.width, world.height)
  world.objects_contacted = function ()
    genome.failed = true
  end
  return world
end

local function destroy(world)
  game_engine.destroy(world)
end

local function update(world, dt)
  if world.failed then return end

  local time_passed = world.time_passed or 0
  if time_passed > 30 then
    world.genome.failed = true
  end
  world.time_passed = time_passed + dt

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
    if v.genome.failed then
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
    for _ = 1, 10 do
      update(v, dt)
    end
  end
end

function love.keypressed(key)
end

local function draw_worlds_in_grid(worlds)
  --local worlds = lume.filter(worlds, function(w) return not (w.genome.failed or false) end)
  local n = #worlds
  if n == 0 then return end

  local sw, sh = love.graphics.getWidth(), love.graphics.getHeight()
  local cw, ch = worlds[1].canvas:getDimensions()

  local bestScale = 0
  local bestCols = 1
  local bestRows = 1

  -- Brute force search for the best grid dimensions
  for cols = 1, n do
    local rows = math.ceil(n / cols)
    local scaleX = sw / (cols * cw)
    local scaleY = sh / (rows * ch)
    local scale = math.min(scaleX, scaleY)

    if scale > bestScale then
      bestScale = scale
      bestCols = cols
      bestRows = rows
    end
  end

  -- Calculate offsets to center the entire grid
  local totalGridW = bestCols * cw * bestScale
  local totalGridH = bestRows * ch * bestScale
  local offsetX = (sw - totalGridW) / 2
  local offsetY = (sh - totalGridH) / 2

  for i, w in ipairs(worlds) do
    local canvas = w.canvas
    local col = (i - 1) % bestCols
    local row = math.floor((i - 1) / bestCols)

    local x = offsetX + (col * cw * bestScale)
    local y = offsetY + (row * ch * bestScale)

    love.graphics.draw(canvas, x, y, 0, bestScale, bestScale)
  end
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
      neat.draw_node_connections(v.genome)
      game_engine.draw(v)
    end)
  end

  draw_worlds_in_grid(worlds)
end
