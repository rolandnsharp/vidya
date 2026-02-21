#!/usr/bin/env bun
/**
 * Convert DailyDialog + foundation data into chat training corpus.
 *
 * Output: one conversation per line, using <|user|> and <|assistant|> turn markers.
 * DailyDialog alternates speakers: odd utterances = user, even = assistant.
 */

import { readFileSync, writeFileSync, existsSync } from 'fs'
import { join, dirname } from 'path'

const dataDir = dirname(new URL(import.meta.url).pathname)

function convertDailyDialog(path) {
  const conversations = []
  const lines = readFileSync(path, 'utf-8').split('\n')

  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed) continue

    const utterances = trimmed.split('__eou__')
      .map(u => u.trim())
      .filter(u => u.length > 0)

    if (utterances.length < 2) continue

    const parts = utterances.map((utt, i) => {
      const role = i % 2 === 0 ? '<|user|>' : '<|assistant|>'
      return `${role} ${utt}`
    })

    conversations.push(parts.join(' '))
  }

  return conversations
}

function loadFoundation(path) {
  return readFileSync(path, 'utf-8')
    .split('\n')
    .map(l => l.trim())
    .filter(l => l.length > 0)
}

// Load DailyDialog (all splits)
let ddConvos = []
for (const split of ['train', 'test', 'validation']) {
  const ddPath = join(dataDir, 'dailydialog', split, `dialogues_${split}.txt`)
  if (existsSync(ddPath)) {
    const convos = convertDailyDialog(ddPath)
    console.log(`DailyDialog ${split}: ${convos.length} conversations`)
    ddConvos.push(...convos)
  }
}

// Load foundation data
const foundation = loadFoundation(join(dataDir, 'foundation.txt'))
console.log(`Foundation: ${foundation.length} conversations`)

// Repeat foundation 20x to give it weight during training
const foundationRepeated = Array(20).fill(foundation).flat()
console.log(`Foundation (repeated): ${foundationRepeated.length} entries`)

// Combine and shuffle (seeded)
const allConvos = [...ddConvos, ...foundationRepeated]

// Simple seeded shuffle
function seededShuffle(arr, seed = 42) {
  let s = seed
  const rng = () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff }
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]]
  }
  return arr
}

seededShuffle(allConvos)

// Write output
const outputPath = join(dataDir, '..', 'chat_input.txt')
writeFileSync(outputPath, allConvos.join('\n') + '\n')

const totalChars = allConvos.reduce((sum, c) => sum + c.length, 0)
console.log(`\nTotal: ${allConvos.length} conversations`)
console.log(`Written to: ${outputPath}`)
console.log(`Total characters: ${totalChars.toLocaleString()}`)
console.log(`Avg conversation length: ${Math.floor(totalChars / allConvos.length)} chars`)
