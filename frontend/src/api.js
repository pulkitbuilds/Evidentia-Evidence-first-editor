import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const client = axios.create({ baseURL: BASE_URL, timeout: 60000 })

// ---------- Cases ----------

export async function listCases() {
  const { data } = await client.get('/cases')
  return data.cases
}

export async function createCase(name) {
  const { data } = await client.post('/cases', { name })
  return data
}

export async function renameCase(caseId, name) {
  const { data } = await client.patch(`/cases/${caseId}`, { name })
  return data
}

export async function deleteCase(caseId) {
  const { data } = await client.delete(`/cases/${caseId}`)
  return data
}

// ---------- Corpus (scoped to a case) ----------

export async function ingestCorpus(caseId, documents) {
  const { data } = await client.post(`/cases/${caseId}/corpus/ingest`, { documents })
  return data
}

export async function corpusStatus(caseId) {
  const { data } = await client.get(`/cases/${caseId}/corpus/status`)
  return data
}

export async function listDocuments(caseId) {
  const { data } = await client.get(`/cases/${caseId}/corpus/documents`)
  return data.documents
}

export async function getDocumentDetail(caseId, docId) {
  const { data } = await client.get(`/cases/${caseId}/corpus/documents/${docId}`)
  return data
}

export async function deleteDocument(caseId, docId) {
  const { data } = await client.delete(`/cases/${caseId}/corpus/documents/${docId}`)
  return data
}

// ---------- Claim checking (scoped to a case) ----------

export async function checkClaim(caseId, sentence, context) {
  const { data } = await client.post(`/cases/${caseId}/claims/check`, { sentence, context })
  return data
}

export async function listHistory(caseId, limit = 100) {
  const { data } = await client.get(`/cases/${caseId}/history/checks`, { params: { limit } })
  return data.checks
}

// ---------- Research assistant (scoped to a case) ----------

export async function startResearch(caseId, topic) {
  const { data } = await client.post(`/cases/${caseId}/research`, { topic })
  return data // { job_id }
}

export async function getResearchJob(caseId, jobId) {
  const { data } = await client.get(`/cases/${caseId}/research/${jobId}`)
  return data
}

export async function listResearchJobs(caseId, limit = 20) {
  const { data } = await client.get(`/cases/${caseId}/research`, { params: { limit } })
  return data.jobs
}

// ---------- Document-level scoring (stateless) ----------

export async function scoreDocument(verdicts) {
  const { data } = await client.post('/document/score', { verdicts })
  return data
}
