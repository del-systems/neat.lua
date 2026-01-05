local NEAT = {}

local default_identity_activation = {
  name = "identity",
  fn = function(x) return x end
}

local default_activations = {
  {
    name = "sigmoid",
    fn = function(x) return 1 / (1 + math.exp(-x)) end
  },
  {
    name = "tanh",
    fn = function(x) return math.tanh(x) end
  },
  {
    name = "relu",
    fn = function(x) return math.max(0, x) end
  },
  {
    name = "leaky_relu",
    fn = function(x) return x > 0 and x or 0.01 * x end
  },
  {
    name = "gaussian",
    fn = function(x) return math.exp(-(x * x)) end
  },
  {
    name = "sin",
    fn = function(x) return math.sin(x) end
  },
  {
    name = "cos",
    fn = function(x) return math.cos(x) end
  },
  {
    name = "abs",
    fn = function(x) return math.abs(x) end
  }
}

local default_output_activations = {
  {
    name = "sigmoid",
    fn = function(x) return 1 / (1 + math.exp(-x)) end
  },
  {
    name = "tanh",
    fn = function(x) return math.tanh(x) end
  }
}

local default_rng = love.math.newRandomGenerator()
local default_elitism_ratio = 0.1
local default_survival_ratio = 0.5
local default_min_fitness = 1
local default_settings = {}

local function get_random_activation(settings)
  local settings = settings or default_settings
  local rng = settings.rng or default_rng
  local activations = settings.activations or default_activations

  return activations[rng:random(#activations)]
end

local function get_random_output_activation(settings)
  local settings = settings or default_settings
  local rng = settings.rng or default_rng
  local activations = settings.activations or default_output_activations

  return activations[rng:random(#activations)]
end

local function get_innovation(settings)
  local settings = settings or default_settings
  if settings.innovation_counter == nil then
    settings.innovation_counter = 0
  end
  settings.innovation_counter = settings.innovation_counter + 1
  return settings.innovation_counter
end

function NEAT.get_default_activation_by_name(name)
  if name == "identity" then
    return default_identity_activation
  end

  for _, v in ipairs(default_activations) do
    if v.name == name then
      return v
    end
  end

  for _, v in ipairs(default_output_activations) do
    if v.name == name then
      return v
    end
  end

  return nil
end

function NEAT.create_genome(settings)
  local settings = settings or default_settings
  local input_count = settings.input_count
  local output_count = settings.output_count
  local output_activations = settings.output_activations or default_output_activations

  local genome = {nodes = {}, connections = {}, fitness = 0, settings = settings}
  for i = 1, input_count do
    table.insert(genome.nodes, {id = i, type = "input", activation = default_identity_activation})
  end
  for i = 1, output_count do
    table.insert(genome.nodes, {id = input_count + i, type = "output", activation = output_activations[1]})
  end

  for i = 1, input_count do
    for j = 1, output_count do
      table.insert(genome.connections, {
        in_node = i,
        out_node = input_count + j,
        weight = 1.0,
        enabled = true,
        innovation = get_innovation(settings)
      })
    end
  end

  return genome
end

-- Crossover: Produce offspring from two parents
function NEAT.crossover(parent1, parent2, settings)
  local settings = settings or default_settings
  local rng = settings.rng or default_rng

  if parent2.fitness > parent1.fitness then parent1, parent2 = parent2, parent1 end

  local child = {nodes = {}, connections = {}, fitness = 0}

  -- Inherit nodes from more fit parent
  for _, node in ipairs(parent1.nodes) do
    child.nodes[node.id] = {id = node.id, type = node.type, activation = node.activation}
  end

  local p2_genes = {}
  for _, conn in ipairs(parent2.connections) do p2_genes[conn.innovation] = conn end

  for _, conn1 in ipairs(parent1.connections) do
    local conn2 = p2_genes[conn1.innovation]
    local gene = {}

    if conn2 and rng:random() > 0.5 then
      for k, v in pairs(conn2) do gene[k] = v end
    else
      for k, v in pairs(conn1) do gene[k] = v end
    end

    if not conn1.enabled or (conn2 and not conn2.enabled) then
      if rng:random() < 0.75 then gene.enabled = false end
    end

    table.insert(child.connections, gene)
  end

  return child
end

-- mutatuin: default, may call all other methods
function NEAT.mutate(genome, settings)
  local settings = settings or default_settings
  local rng = settings.rng or default_rng

  -- weight mutation
  for _, conn in ipairs(genome.connections) do
    if rng:random() < 0.8 then
      conn.weight = conn.weight + (rng:random() * 2 - 1) * 0.5
    else
      conn.weight = rng:random() * 2 - 1
    end
  end

  -- activation mutation
  if rng:random() < 0.1 then
    local node = genome.nodes[rng:random(#genome.nodes)]
    if node.type == "hidden" then
      node.activation = get_random_activation(settings)
    elseif node.type == "output" then
      node.activation = get_random_output_activation(settings)
    end
  end

  -- structural mutations
  if rng:random() < 0.05 then NEAT.mutate_add_connection(genome, settings) end
  if rng:random() < 0.1 then NEAT.mutate_add_node(genome, settings) end

  if rng:random() < 0.05 then -- 5% chance to attempt to toggle connection, helps with disabled 'viruses'
    NEAT.mutate_toggle_connection(genome, settings)
  end
end

-- combat disabling all connections if parents got disabled by accident
-- as it may result all next generation connections to be disabled
function NEAT.mutate_toggle_connection(genome, settings)
  local settings = settings or default_settings
  local rng = settings.rng or default_rng

  if #genome.connections > 0 then
    local conn = genome.connections[rng:random(1, #genome.connections)]
    conn.enabled = not conn.enabled
  end
end

-- mutation: add a new connection between existing nodes
function NEAT.mutate_add_connection(genome, settings)
  local settings = settings or default_settings
  local rng = settings.rng or default_rng

  -- attempt up to 20 times to find a valid new connection
  for _ = 1, 20 do
    local n1 = genome.nodes[rng:random(#genome.nodes)]
    local n2 = genome.nodes[rng:random(#genome.nodes)]

    -- rule: cannot connect to an input or from an output (standard feed-forward)
    -- rule: avoid connecting a node to itself
    local is_valid = n1.id ~= n2.id and n2.type ~= "input" and n1.type ~= "output"

    if is_valid then
      -- Check if connection already exists
      local exists = false
      for _, conn in ipairs(genome.connections) do
        if conn.in_node == n1.id and conn.out_node == n2.id then
          exists = true
          break
        end
      end

      if not exists then
        table.insert(genome.connections, {
          in_node = n1.id,
          out_node = n2.id,
          weight = rng:random() * 2 - 1,
          enabled = true,
          innovation = get_innovation(settings)
        })

        return true
      end
    end
  end
  return false
end

-- mutation: add a new node by splitting an existing connection
function NEAT.mutate_add_node(genome, settings)
  local settings = settings or default_settings
  local rng = settings.rng or default_rng

  if #genome.connections == 0 then return false end

  -- pick an enabled connection to split
  local enabled_conns = {}
  for i, conn in ipairs(genome.connections) do
    if conn.enabled then table.insert(enabled_conns, i) end
  end

  if #enabled_conns == 0 then return false end
  local index = enabled_conns[rng:random(#enabled_conns)]
  local old_conn = genome.connections[index]

  -- disable old connection
  old_conn.enabled = false

  -- create new hidden node
  local new_node_id = #genome.nodes + 1
  table.insert(genome.nodes, {
    id = new_node_id,
    type = "hidden",
    activation = get_random_activation(settings)
  })

  -- add connection: in -> new (weight 1.0)
  table.insert(genome.connections, {
    in_node = old_conn.in_node,
    out_node = new_node_id,
    weight = 1.0,
    enabled = true,
    innovation = get_innovation(settings)
  })

  -- add connection: new -> out (original weight)
  table.insert(genome.connections, {
    in_node = new_node_id,
    out_node = old_conn.out_node,
    weight = old_conn.weight,
    enabled = true,
    innovation = get_innovation(settings)
  })

  return true
end

-- generation management: evolution step
function NEAT.evolve_population(population, settings)
  local settings = settings or default_settings
  local rng = settings.rng or default_rng

  local next_gen = {}

  -- shallow copy only known keys as user may add own variables (like world, texture, engine and etc)
  for k, v in ipairs(population) do
    table.insert(next_gen, {
      nodes = v.nodes,
      connections = v.connections,
      fitness = v.fitness,
      settings = v.settings
    })
  end

  if #next_gen >= 2 then
    -- 1. sort by fitness (descending)
    table.sort(next_gen, function(a, b) return a.fitness > b.fitness end)

    -- 2. elitism: keep top performers as it is
    local elitism_ratio = settings.elitism_ratio or default_elitism_ratio
    local elitism_count = math.max(1, math.floor(#next_gen * elitism_ratio))

    -- 3. survived genes should mutate or crossover
    local survival_ratio = settings.survival_ratio or default_survival_ratio
    local survivors_count = math.max(1, math.floor(#next_gen * survival_ratio))

    -- 4. remaining ones are crossovers and mutations
    if elitism_count + survivors_count <= #next_gen then
      for i = elitism_count + survivors_count, #next_gen do
        local p1 = population[rng:random(1, elitism_count)]
        local p2 = population[rng:random(1, elitism_count + survivors_count)]
        local child = NEAT.crossover(p1, p2, settings)

        NEAT.mutate(child, settings)
        next_gen[i] = child
      end
    end

    -- 5. heal 'disabled' viruses. toggle connection for a random chosen one
    local r = population[rng:random(elitism_count + survivors_count, #next_gen)]
    NEAT.mutate_toggle_connection(r, settings)
  end

  -- 6. reset fitness values for the next values
  for _, g in ipairs(next_gen) do
    -- 7. mutate every genome if its low
    if g.fitness < (settings.min_fitness or default_min_fitness) then
      NEAT.mutate(g, settings)
    end
    g.fitness = 0
  end

  return next_gen
end

-- evaluation
function NEAT.evaluate(genome, inputs, settings)
  local values = {}
  for i, v in ipairs(inputs) do values[genome.nodes[i].id] = v end

  -- sort connections by innovation to approximate feed-forward order
  table.sort(genome.connections, function(a, b) return a.innovation < b.innovation end)

  for _, conn in ipairs(genome.connections) do
    if conn.enabled and values[conn.in_node] then
      local node = genome.nodes[conn.out_node]
      local input = values[conn.in_node] * conn.weight
      if not node then
        print('WTF')
        NEAT.print_genome(genome)
      end
      values[conn.out_node] = (node.activation.fn or NEAT.get_default_activation_by_name(node.activation.name))( (values[conn.out_node] or 0) + input )
    end
  end

  local results = {}
  for _, node in ipairs(genome.nodes) do
    if node.type == "output" then table.insert(results, values[node.id] or 0) end
  end

  local return_values = {}
  for i, v in ipairs(results) do
    table.insert(return_values, v)
  end

  return return_values
end

-- make a copy of a genome purely transferable between threads and serialization
-- be aware that custom activations and rng seed is lost
function NEAT.purify_genome(genome)
  local copy = {
    fitness = genome.fitness,
    settings = {},
    nodes = {},
    connections = {}
  }

  copy.settings.innovation_counter = genome.settings.innovation_counter
  for _, node in ipairs(genome.nodes) do
    table.insert(copy.nodes, { id = node.id, type = node.type, activation = { name = node.activation.name } })
  end

  for _, conn in ipairs(genome.connections) do
    table.insert(copy.connections, {
      in_node = conn.in_node,
      out_node = conn.out_node,
      weight = conn.weight,
      innovation = conn.innovation,
      enabled = conn.enabled
    })
  end

  return copy
end

-- human-readable text summary of the genome
function NEAT.print_genome(genome)
    print("--- Genome Report ---")
    print(string.format("Fitness: %.2f", genome.fitness))
    print("Nodes:")
    for _, node in ipairs(genome.nodes) do
        print(string.format("  [%d] Type: %s, Activation: %s",
            node.id, node.type, node.activation.name))
    end
    print("Connections:")
    for _, conn in ipairs(genome.connections) do
        local status = conn.enabled and "Enabled" or "Disabled"
        print(string.format("  In:%d -> Out:%d | Weight: %.4f | Innov: %d | [%s]",
            conn.in_node, conn.out_node, conn.weight, conn.innovation, status))
    end
    print("---------------------")
end

-- generate mermaid graph syntax
function NEAT.to_mermaid(genome)
  local lines = {"graph LR"}

  -- define node styles based on type
  for _, node in ipairs(genome.nodes) do
    local style = ""
    if node.type == "input" then
      style = string.format("%d((In %d))", node.id, node.id)
    elseif node.type == "output" then
      style = string.format("%d{{Out %d: %s}}", node.id, node.id, node.activation.name)
    else
      style = string.format("%d([Hidden %d: %s])", node.id, node.id, node.activation.name)
    end
    table.insert(lines, "    " .. style)
  end

  -- define connections
  for _, conn in ipairs(genome.connections) do
    if conn.enabled then
      -- format weight to 2 decimal places
      local weight_label = string.format("%.2f", conn.weight)
      -- use thicker lines for stronger weights
      local arrow = " -- " .. weight_label .. " --> "
      table.insert(lines, string.format("    %d%s%d",
      conn.in_node, arrow, conn.out_node))
    else
      -- represent disabled connections as dotted lines
      -- table.insert(lines, string.format("    %d -. disabled .-> %d",
      -- conn.in_node, conn.out_node))
    end
  end

  return table.concat(lines, "\n")
end

-- draw the node structure
function NEAT.draw_node_connections(genome)
  local width = love.graphics.getWidth()
  local height = love.graphics.getHeight()
  local canvas = love.graphics.getCanvas()
  if canvas then
    width, height = canvas:getDimensions()
  end
  local padding_x = 50
  local padding_y = 50
  local node_radius = 6

  -- 1. Map nodes by ID for quick lookup
  local node_map = {}
  local inputs = {}
  local outputs = {}
  local hidden = {}

  for _, node in ipairs(genome.nodes) do
    node_map[node.id] = node
    -- Initialize layout data
    node._depth = 0
    node._layer_idx = 0

    if node.type == "input" then table.insert(inputs, node)
    elseif node.type == "output" then table.insert(outputs, node)
    else table.insert(hidden, node) end
  end

  -- 2. Calculate Depth (Layering)
  -- Inputs are always depth 0. Propagate depth to hidden/output nodes.
  -- We loop enough times to propagate through the network.
  local max_depth = 1
  local iterations = #genome.nodes + 2 -- Cap iterations to prevent infinite loops from recurrent connections

  for i = 1, iterations do
    local changed = false
    for _, conn in ipairs(genome.connections) do
      if conn.enabled then
        local n_in = node_map[conn.in_node]
        local n_out = node_map[conn.out_node]

        if n_in and n_out then
          -- If flow is forward (avoid pushing depth for recurrent links if possible,
          -- though simple recurrence will just hit the iteration cap)
          if n_out.type ~= "input" then
            if n_out._depth < n_in._depth + 1 then
              n_out._depth = n_in._depth + 1
              changed = true
              if n_out._depth > max_depth then max_depth = n_out._depth end
            end
          end
        end
      end
    end
    if not changed then break end
  end

  -- Force outputs to the far right (max_depth) for visual clarity
  for _, node in ipairs(outputs) do
    node._depth = max_depth
  end

  -- 3. Group nodes by depth to calculate Y positions
  local layers = {}
  for i = 0, max_depth do layers[i] = {} end

  -- Add nodes to their respective layer buckets
  for _, node in ipairs(genome.nodes) do
    -- Safety clamp in case of detached nodes
    local d = math.min(node._depth, max_depth)
    table.insert(layers[d], node)
  end

  -- 4. Assign Coordinates
  for d = 0, max_depth do
    local layer_nodes = layers[d]
    -- Sort by ID to keep consistent vertical order between frames
    table.sort(layer_nodes, function(a,b) return a.id < b.id end)

    local count = #layer_nodes
    local col_x = padding_x + (d / max_depth) * (width - 2 * padding_x)

    for i, node in ipairs(layer_nodes) do
      node.x = col_x
      -- Distribute vertically
      node.y = padding_y + (i - 0.5) * ((height - 2 * padding_y) / math.max(count, 1))
    end
  end

  -- 5. Draw Connections
  for _, conn in ipairs(genome.connections) do
    if conn.enabled then
      local n1 = node_map[conn.in_node]
      local n2 = node_map[conn.out_node]

      if n1 and n2 then
        -- Visual settings
        if conn.weight > 0 then love.graphics.setColor(0, 1, 0, 1)
        else love.graphics.setColor(1, 0, 0, 1) end

        love.graphics.setLineWidth(math.max(1, math.abs(conn.weight) * 2))
        love.graphics.line(n1.x, n1.y, n2.x, n2.y)

        -- Draw Weight Label
        local mid_x = (n1.x + n2.x) / 2
        local mid_y = (n1.y + n2.y) / 2
        local weight_str = string.format("%.2f", conn.weight)
        local font = love.graphics.getFont()
        local text_w = font:getWidth(weight_str)
        local text_h = font:getHeight()

        -- Background box for text
        love.graphics.setColor(0, 0, 0, 0.7)
        love.graphics.rectangle("fill", mid_x - text_w/2 - 2, mid_y - text_h/2 - 2, text_w + 4, text_h + 4)

        -- Text
        love.graphics.setColor(1, 1, 1, 1)
        love.graphics.print(weight_str, mid_x - text_w/2, mid_y - text_h/2)
      end
    end
  end

  -- 6. Draw Nodes
  for _, node in ipairs(genome.nodes) do
    -- Node circle
    love.graphics.setColor(0.1, 0.1, 0.1)
    love.graphics.circle("fill", node.x, node.y, node_radius)

    -- Node outline (Color by type)
    if node.type == "input" then love.graphics.setColor(0.2, 0.6, 1)      -- Blue
    elseif node.type == "output" then love.graphics.setColor(1, 0.6, 0.2) -- Orange
    else love.graphics.setColor(1, 1, 1) end                              -- White

    love.graphics.setLineWidth(2)
    love.graphics.circle("line", node.x, node.y, node_radius)

    -- Node Label
    love.graphics.setColor(1, 1, 1)
    love.graphics.print(node.activation.name or node.id, node.x + node_radius + 4, node.y - 8)
  end
end
return NEAT
