
/**
 * script.js
 * This file contains the JavaScript logic for the Battleship game web interface.
 * It handles game initialization, user interactions, and rendering of game boards.
 */
    
  let autoMoveInterval = null;
  /**
   * Indicates whether the game is in manual ship placement phase.
   *
   * @type {boolean}
   */
  let manualPhase = false;
  /**
   * Holds the starting cell for manual ship placement.
   *
   * @type {*}
   */
  let manualStart = null;
  /**
   * Chart instance for displaying game statistics.
   *
   * @type {*}
   */
  let statsChart = null;
  

  /**
   * Toggle the visibility of the boat placement options
   *
   * @type {*}
   */
  const boatPlacementContainer = document.getElementById('boatPlacementContainer');

  /** Updates the boat placement visibility */
  function updateBoatPlacementVisibility() {
    const mode = document.getElementById('gameMode').value;
    const placementSection = document.getElementById('placementOptions');
    // show only for User vs MCTS
    placementSection.style.display = (mode === 'uservsmcts') ? 'block' : 'none';
  }

  window.addEventListener('load', updateBoatPlacementVisibility);
  document.getElementById('gameMode').addEventListener('change', updateBoatPlacementVisibility);

  // --- Tab switching logic for statistics ---
  document.querySelectorAll('.tabButton').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('.tabButton').forEach(x => x.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll('.tabContent').forEach(c => c.style.display = 'none');
      const content = document.getElementById(btn.dataset.tab);
      if (btn.dataset.tab === 'dashboardTab') {
        content.style.display = 'flex';  
      } else {
        content.style.display = 'block';
      }
    };
  });

  /**
   * Renders the summary of AI movements
   *
   * @param {*} summary 
   */
  function renderSummary(summary) {
    const container = document.getElementById('explanationText');
      if (!summary || !summary.length) {
        container.textContent = 'No movement returned by the AI.';
        return;
      }
  
    const movesList = summary
      .map(r => `(${r.action.x},${r.action.y})`)
      .join(', ')
      .replace(/, ([^,]*)$/, ' y $1');

    const best = summary.reduce((a, b) => a.visits > b.visits ? a : b);

    container.textContent = 
      `Evaluated movements: ${movesList}. `
      + `Most explored: (${best.action.x},${best.action.y}) with `
      + `${best.visits} visits and ${best.wins} victories `
      + `(ratio ${best.win_rate.toFixed(2)}).`;
  }

  /**
   * Converts a tree structure into Vis.js data format
   *
   * @param {*} node 
   * @param {{}} [nodes=[]] 
   * @param {{}} [edges=[]] 
   * @param {*} [parentId=null] 
   * @returns {{ nodes: {}; edges: {}; }} 
   */
  function treeToVisData(node, nodes = [], edges = [], parentId = null) {
    const id = nodes.length;
    nodes.push({ id, label: `${node.action}\nV=${node.visits}\nW=${node.wins}` });
    if (parentId !== null) edges.push({ from: parentId, to: id });
    (node.children || []).forEach(c =>
      treeToVisData(c, nodes, edges, id)
    );
    return { nodes, edges };
  }

  /**
   * Renders a tree structure using Vis.js
   *
   * @param {*} tree 
   */
  function renderVisTree(tree) {
    const { nodes, edges } = treeToVisData(tree);
    const container = document.getElementById('network');
    container.innerHTML = '';
    new vis.Network(container, { nodes, edges }, {
      layout: { hierarchical:{ enabled:true, direction:'UD', sortMethod:'directed', levelSeparation:100, nodeSpacing:100 } },
      physics: false,
      edges: { smooth:{ type:'cubicBezier', forceDirection:'horizontal', roundness:0.4 } },
      nodes: { shape:'ellipse' }
    });
  }

  /**
   * Draws the game board in the specified container
   *
   * @param {*} board 
   * @param {*} containerId 
   * @param {*} clickable 
   * @param {*} boardType 
   */
  function drawBoard(board, containerId, clickable, boardType) {
    let html = '<table>';
    for (let i = 0; i < board.length; i++) {
      html += '<tr>';
      for (let j = 0; j < board[i].length; j++) {
        const cell = board[i][j];
        // start with 'clickable' if needed
        let cls = clickable ? 'clickable' : '';

        // If the cell is a number → ship segment
        if (/^[0-9]+$/.test(cell)) {
          cls += ' ship-cell';
        }
        // If it's a hit
        else if (cell === 'X') {
          cls += ' hit';
        }
        // If it's a miss
        else if (cell === 'O') {
          cls += ' miss';
        }
        // Otherwise it's empty water
        else {
          cls += ' empty';
        }

        html += `<td class="${cls.trim()}" data-i="${i}" data-j="${j}" data-board="${boardType}"></td>`;
      }
      html += '</tr>';
    }
    html += '</table>';
    document.getElementById(containerId).innerHTML = html;

    // Attach click handlers for clickable boards
    if (clickable) {
      document.querySelectorAll(`#${containerId} td.clickable`).forEach(cell => {
        cell.onclick = () => {
          const x  = +cell.dataset.i;
          const y  = +cell.dataset.j;
          const bt = cell.dataset.board;

          if (manualPhase && bt === 'user') {
            // Manual ship placement phase
            if (!manualStart) {
              manualStart = { x, y };
              document.getElementById('message').innerText =
                `Start cell set at (${x},${y}). Now click end cell.`;
            } else {
              manualPlace(manualStart, { x, y });
              manualStart = null;
            }
          }
          else if (bt === 'pc') {
            cellClicked(x, y);
          }
        };
      });
    }
  }
    

  /**
   * Updates the status of user and PC boats
   *
   * @param {*} userBoats 
   * @param {*} pcBoats 
   */
  function updateBoatStatus(userBoats, pcBoats) {
    const makeList = boats => boats.map(b =>
      `<li${b.sunk ? ' class="sunk"' : ''}>Ship ${b.ship}</li>`
    ).join('');
    document.getElementById('userBoatsStatus').innerHTML = `<ul>${makeList(userBoats)}</ul>`;
    document.getElementById('pcBoatsStatus').innerHTML   = `<ul>${makeList(pcBoats)}</ul>`;
  }


  /**
   * Handles cell clicks on the game board 
   *
   * @param {*} x 
   * @param {*} y 
   */
    function cellClicked(x, y) {
    const mode  = document.getElementById('gameMode').value;
    const msgEl = document.getElementById('message');

    fetch('/user_move', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ x, y })
    })
    .then(r => r.json())
    .then(d => {
      drawBoard(d.pc_board, 'pcBoard', true, 'pc');
      updateBoatStatus(d.user_boats, d.pc_boats);

      if (mode === 'uservsmcts' && d.summary) {
        msgEl.innerText = 'AI is thinking…';
        setTimeout(() => applyResponse(d), 500);
      } else {
        applyResponse(d);
      }
    })
    .catch(console.error);

    function applyResponse(d) {
      document.getElementById('message').innerText = d.message;
      drawBoard(d.user_board, 'userBoard', false, 'user');
      drawBoard(d.pc_board,   'pcBoard',   true,  'pc');
      updateBoatStatus(d.user_boats, d.pc_boats);

      if (mode === 'uservsmcts' && d.summary) {
        renderSummary(d.summary);
        renderVisTree(d.tree);
        document.getElementById('treeContainer').style.display = 'block';
      }
    }
  }


  // Place a ship manually
  /**
   * Manually places a ship on the board
   *
   * @param {*} start 
   * @param {*} end 
   */
  function manualPlace(start, end) {
    fetch('/manual_place', {
      method: 'POST',
      headers: { 'Content-Type':'application/json' },
      body: JSON.stringify({ start, end })
    })
    .then(r => r.json())
    .then(d => {
      document.getElementById('message').innerText = d.message;
      drawBoard(d.user_board, 'userBoard', d.manual_phase, 'user');
      updateBoatStatus(d.user_boats, []);
      manualPhase = d.manual_phase;
    });
  }

  /** 
   * Starts a new game by sending a request to the server
   */
  function startGame() {
    fetch('/start',{ method:'POST' })
    .then(r => r.json())
    .then(d => {
      manualPhase = d.manual_phase;
      document.getElementById('message').innerText = d.message;
      drawBoard(d.user_board, 'userBoard', manualPhase, 'user');
      drawBoard(d.pc_board,   'pcBoard',   true,       'pc');
      updateBoatStatus(d.user_boats, d.pc_boats);

      if (document.getElementById('gameMode').value === 'mcts_vs_ml_mcts') {
        // Auto-play for AI vs AI
        document.getElementById('userBoardHeading').innerText = 'MCTS Board';
        document.getElementById('pcBoardHeading').innerText   = 'ML-MCTS Board';
        document.getElementById('userBoatsHeading').innerText = 'MCTS Ships';
        document.getElementById('pcBoatsHeading').innerText   = 'ML-MCTS Ships';
        clearInterval(autoMoveInterval);
        autoMoveInterval = setInterval(autoMove, 1000);
      } else {
        document.getElementById('userBoardHeading').innerText = 'Your Board';
        document.getElementById('pcBoardHeading').innerText   = 'PC Board';
        document.getElementById('userBoatsHeading').innerText = 'Your Boats';
        document.getElementById('pcBoatsHeading').innerText   = 'PC Boats';
        clearInterval(autoMoveInterval);
      }

      // clear the explanation text
      const explan = document.getElementById('explanationText');
      if (explan) explan.textContent = '';

      // clenar the network visualization
      const net = document.getElementById('network');
      if (net) net.innerHTML = '';
    });
  }

  /** 
   * Automatically moves the AI players in MCTS vs ML-MCTS mode
   * by sending a request to the server.
   */
  function autoMove() {
    fetch('/auto_move',{ method:'POST' })
    .then(r => r.json())
    .then(d => {
      document.getElementById('message').innerText = d.message;
      drawBoard(d.user_board, 'userBoard', false, 'user');
      drawBoard(d.pc_board,   'pcBoard',   false, 'pc');
      updateBoatStatus(d.user_boats, d.pc_boats);
      if (d.game_over) clearInterval(autoMoveInterval);
    });
  }


  /** 
   * Sets the game options based on user selection
   * and starts the game.
   */
  function setGameOptions() {
    const mode = document.getElementById('gameMode').value;
    const pcClick  = (mode === 'uservsmcts');
    let placement = 'random';
    if (mode === 'uservsmcts') {
      document.getElementsByName('boatPlacement').forEach(radio => {
        if (radio.checked) placement = radio.value;
      });
    }

    fetch('/set_options',{
      method: 'POST',
      headers: { 'Content-Type':'application/json' },
      body: JSON.stringify({ game_mode: mode, boat_placement: placement })
    })
    .then(r => r.json())
    .then(d => {
      manualPhase = d.manual_phase;

      if (mode === 'statistics') {
        // Show statistics view
        document.getElementById('options-container').style.display = 'none';
        document.getElementById('gameContainer').style.display    = 'none';
        document.getElementById('statsContainer').style.display   = 'block';
        updateStats();
      } else {
        // Show game view
        document.getElementById('options-container').style.display = 'none';
        document.getElementById('statsContainer').style.display   = 'none';
        document.getElementById('gameContainer').style.display    = 'block';

        document.getElementById('message').innerText = d.message;
        drawBoard(d.user_board,'userBoard',d.manual_phase,'user');
        drawBoard(d.pc_board,   'pcBoard',   pcClick,'pc');
        updateBoatStatus(d.user_boats, d.pc_boats);

        if (mode === 'mcts_vs_ml_mcts') {
          document.getElementById('userBoardHeading').innerText = 'MCTS Board';
          document.getElementById('pcBoardHeading').innerText   = 'ML-MCTS Board';
          document.getElementById('userBoatsHeading').innerText = 'MCTS Ships';
          document.getElementById('pcBoatsHeading').innerText   = 'ML-MCTS Ships';
          clearInterval(autoMoveInterval);
          autoMoveInterval = setInterval(autoMove, 1000);
        } else {
          document.getElementById('userBoardHeading').innerText = 'Your Board';
          document.getElementById('pcBoardHeading').innerText   = 'PC Board';
          document.getElementById('userBoatsHeading').innerText = 'Your Boats';
          document.getElementById('pcBoatsHeading').innerText   = 'PC Boats';
          clearInterval(autoMoveInterval);
        }
      }
    });
  }


  /** 
   * Returns to the main menu, clearing the game state
   * and stopping any ongoing intervals.
   */
  function backToMenu() {
    clearInterval(autoMoveInterval);
    manualPhase = false;
    manualStart = null;
    document.getElementById('options-container').style.display = 'block';
    document.getElementById('gameContainer').style.display    = 'none';
    document.getElementById('statsContainer').style.display   = 'none';
  }


/**
 * Fetches and updates game statistics from the server
 * 
 */
function updateStats() {
  fetch('/stats')
    .then(res => res.json())
    .then(data => {
      // — Tables 2 cols —
      const dt1 = $('#statsTable1').DataTable();
      dt1.clear();
      data.uservsmcts.forEach(r => {
        dt1.row.add([ r.winner, r.duration.toFixed(2) ]);
      });
      dt1.draw();

      const dt2 = $('#statsTable2').DataTable();
      dt2.clear();
      data.mcts_vs_ml_mcts.forEach(r => {
        dt2.row.add([ r.winner, r.duration.toFixed(2) ]);
      });
      dt2.draw();

      // — KPI Dashboard —
      const arr1     = data.uservsmcts;
      const total1   = arr1.length;
      const winsUser = arr1.filter(r => r.winner === 'user').length;
      const winsMCTS = arr1.filter(r => r.winner === 'MCTS').length;
      const dursUser = arr1.filter(r => r.winner === 'user').map(r => r.duration);
      const dursMCTS = arr1.filter(r => r.winner === 'MCTS').map(r => r.duration);

      document.getElementById('kpi-total-1').textContent         = total1 || '–';
      document.getElementById('kpi-wpct-user-1').textContent     = total1
        ? ((winsUser/total1)*100).toFixed(1) + '%'
        : '–';
      document.getElementById('kpi-wpct-mcts-1').textContent     = total1
        ? ((winsMCTS/total1)*100).toFixed(1) + '%'
        : '–';
      document.getElementById('kpi-avgdur-user-1').textContent   = dursUser.length
        ? (dursUser.reduce((s,v)=>s+v,0)/dursUser.length).toFixed(2)
        : '–';
      document.getElementById('kpi-avgdur-mcts-1').textContent   = dursMCTS.length
        ? (dursMCTS.reduce((s,v)=>s+v,0)/dursMCTS.length).toFixed(2)
        : '–';

      const arr2      = data.mcts_vs_ml_mcts;
      const total2    = arr2.length;
      const winsMCTS2 = arr2.filter(r => r.winner === 'MCTS').length;
      const winsML    = arr2.filter(r =>
        r.winner === 'NeuralMCTS' || r.winner === 'ML-MCTS'
      ).length;
      const dursMCTS2 = arr2.filter(r => r.winner === 'MCTS').map(r => r.duration);
      const dursML    = arr2.filter(r =>
        r.winner === 'NeuralMCTS' || r.winner === 'ML-MCTS'
      ).map(r => r.duration);

      document.getElementById('kpi-total-2').textContent         = total2 || '–';
      document.getElementById('kpi-wpct-mcts-2').textContent     = total2
        ? ((winsMCTS2/total2)*100).toFixed(1) + '%'
        : '–';
      document.getElementById('kpi-wpct-ml-2').textContent       = total2
        ? ((winsML/total2)*100).toFixed(1) + '%'
        : '–';
      document.getElementById('kpi-avgdur-mcts-2').textContent   = dursMCTS2.length
        ? (dursMCTS2.reduce((s,v)=>s+v,0)/dursMCTS2.length).toFixed(2)
        : '–';
      document.getElementById('kpi-avgdur-ml-2').textContent     = dursML.length
        ? (dursML.reduce((s,v)=>s+v,0)/dursML.length).toFixed(2)
        : '–';

      // — Chart.js: % of wins per player —
      const labels  = ['User vs MCTS', 'MCTS vs ML-MCTS'];
      const pctUser = [
        total1 ? (winsUser/total1*100).toFixed(1) : 0,
        0
      ];
      const pctMCTS = [
        total1 ? (winsMCTS/total1*100).toFixed(1) : 0,
        total2 ? (winsMCTS2/total2*100).toFixed(1) : 0
      ];
      const pctML   = [
        0,
        total2 ? (winsML/total2*100).toFixed(1) : 0
      ];

      const ctx = document.getElementById('statsChart').getContext('2d');
      if (statsChart) statsChart.destroy();
      statsChart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [
            { label: 'User wins %',    data: pctUser, backgroundColor: 'rgba(54,162,235,0.6)' },
            { label: 'MCTS wins %',    data: pctMCTS, backgroundColor: 'rgba(255,99,132,0.6)' },
            { label: 'ML-MCTS wins %', data: pctML,   backgroundColor: 'rgba(255,206,86,0.6)' }
          ]
        },
        options: {
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              ticks: { callback: v => v + '%' }
            }
          },
          plugins: {
            legend: { position: 'top' },
            tooltip: {
              callbacks: {
                label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y}%`
              }
            }
          }
        }
      });
    })
    .catch(err => console.error('Failed to load stats:', err));
}



  /**
   * Exports the game statistics to a CSV file.
   *
   * @type {*}
   */
  const exportBtn = document.getElementById('exportCsvBtn');
  if (exportBtn) {
    exportBtn.addEventListener('click', () => {
      fetch('/stats')
        .then(res => {
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          return res.json();
        })
        .then(data => {
          let csv = 'mode,winner,duration\n';
          data.uservsmcts.forEach(row => {
            csv += `uservsmcts,${row.winner},${row.duration}\n`;
          });
          data.mcts_vs_ml_mcts.forEach(row => {
            csv += `mcts_vs_ml_mcts,${row.winner},${row.duration}\n`;
          });

          const blob = new Blob([csv], { type: 'text/csv' });
          const url  = URL.createObjectURL(blob);
          const a    = document.createElement('a');
          a.href     = url;
          a.download = 'battleship_stats.csv';
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
        })
        .catch(err => console.error('Failed to export CSV:', err));
    });
  }

  // Initialize DataTables on page ready
  $(document).ready(function(){
    $('#statsTable1').DataTable({ paging: true, info: false });
    $('#statsTable2').DataTable({ paging: true, info: false });
  });

  /**
   * Link to the Help section in the navigation bar
   *
   * @type {*}
   */
  const helpLink    = document.getElementById('nav-help');

  /**
   * Dropdown menu for Help section
   *
   * @type {*}
   */
  const helpDropdown= document.getElementById('helpDropdown');

  /**
   * Button to toggle the About section
   *
   * @type {*}
   */
  const aboutBtn    = document.getElementById('btn-about');

  /**
   * Button to restart the game and return to the main menu
   *
   * @type {*}
   */
  const restartBtn  = document.getElementById('btn-restart');

  /**
   * Line that displays information about the game
   *
   * @type {*}
   */
  const aboutLine   = document.getElementById('aboutLine');

  // Toggle dropdown when clicking Help
  helpLink.addEventListener('click', e => {
    e.preventDefault();
    helpDropdown.style.display =
      (helpDropdown.style.display === 'block') ? 'none' : 'block';
    aboutLine.style.display = 'none';
  });

  // Close dropdown when clicking outside
  document.addEventListener('click', e => {
    if (!helpLink.contains(e.target) && !helpDropdown.contains(e.target)) {
      helpDropdown.style.display = 'none';
    }
  });

  aboutBtn.addEventListener('click', () => {
    aboutLine.style.display = 
      (aboutLine.style.display === 'none') ? 'block' : 'none';
  });

  // Restart — back to main menu
  restartBtn.addEventListener('click', () => {
    helpDropdown.style.display = 'none';
    backToMenu();
  });

  // Home - Main Menu
  document.getElementById('nav-home').addEventListener('click', e => {
    e.preventDefault();
    if (helpDropdown) helpDropdown.style.display = 'none';
    backToMenu();
  });

  // Statistics - show stats view
  document.getElementById('nav-stats').addEventListener('click', e => {
    e.preventDefault();
    if (helpDropdown) helpDropdown.style.display = 'none';
    document.getElementById('options-container').style.display = 'none';
    document.getElementById('gameContainer').style.display    = 'none';
    document.getElementById('statsContainer').style.display   = 'block';
    updateStats();
  });

  // Play - start a new game with the current selected values
  document.getElementById('nav-play').addEventListener('click', e => {
    e.preventDefault();
    if (helpDropdown) helpDropdown.style.display = 'none';
    setGameOptions();
  });



  