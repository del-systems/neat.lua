local game_engine = {}

function game_engine.new_world()
  local world = {
    width = 640,
    height = 480,
    meter = 64
  }

  world.physics_world = love.physics.newWorld(0, 9.81 * world.meter, false)

  world.platform = {}
  world.platform.body = love.physics.newBody(world.physics_world, world.width / 2, world.height / 2, "kinematic")
  world.platform.shape = love.physics.newRectangleShape(100, 30)
  world.platform.fixture = love.physics.newFixture(world.platform.body, world.platform.shape)

  world.stick = {}
  world.stick.body = love.physics.newBody(world.physics_world, world.width / 2, world.height / 2 - 50 - 30, "dynamic")
  world.stick.shape = love.physics.newRectangleShape(10, 100)
  world.stick.fixture = love.physics.newFixture(world.stick.body, world.stick.shape)
  world.stick.body:applyForce(love.math.random(-1000, 1000), 0)

  world.joint = love.physics.newRevoluteJoint(world.platform.body, world.stick.body, world.width / 2, world.height / 2, true)

  world.physics_world:setCallbacks(function(fixtureA, fixtureB)
    if (fixtureA == world.platform.fixture and fixtureB == world.stick.fixture) or
      (fixtureA == world.stick.fixture and fixtureB == world.platform.fixture) then
      world.failed = true
    end
  end)

  return world
end

function game_engine.update(world, dt)
  world.physics_world:update(dt)
end

function game_engine.draw(world)
  love.graphics.setColor(0.28, 0.63, 0.05)
  love.graphics.polygon("fill", world.platform.body:getWorldPoints(world.platform.shape:getPoints()))

  love.graphics.setColor(0.76, 0.18, 0.05)
  love.graphics.polygon("fill", world.stick.body:getWorldPoints(world.stick.shape:getPoints()))
end

function game_engine.destroy(world)
  world.physics_world:destroy()
end

return game_engine
