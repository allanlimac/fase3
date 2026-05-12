# Guia de Uso Seguro — WomenHealth-LLM

## Identificação do Modelo

| Campo | Valor |
|---|---|
| **Nome** | WomenHealth-LLM |
| **Finalidade** | Assistência informacional em saúde da mulher |
| **Domínio** | Ginecologia, obstetrícia, saúde reprodutiva, violência doméstica, saúde mental feminina |
| **Idioma** | Português brasileiro |
| **Técnica de treinamento** | Supervised Fine-Tuning (SFT) com LoRA/QLoRA |
| **Validação** | Requer aprovação por profissional de saúde qualificado antes do uso clínico |

---

## O que este modelo PODE fazer

- Responder perguntas frequentes sobre saúde da mulher em linguagem acessível
- Explicar sintomas, exames laboratoriais, exames de imagem e seus laudos
- Descrever procedimentos ginecológicos e obstétricos comuns
- Identificar sinais de alerta que requerem atendimento urgente
- Orientar sobre quando procurar atendimento médico
- Apoiar a triagem inicial de sintomas (sem substituir avaliação clínica)
- Fornecer informações educativas sobre planejamento familiar e saúde reprodutiva
- Acolher e orientar mulheres em situação de violência doméstica
- Explicar o ciclo menstrual, menopausa e climatério
- Fornecer informações sobre amamentação e cuidados no puerpério

## O que este modelo NÃO pode fazer

- Fornecer diagnóstico médico definitivo
- Prescrever medicamentos, doses ou tratamentos individualizados
- Substituir a consulta com médico, ginecologista, obstetra ou qualquer profissional de saúde
- Interpretar exames clínicos específicos do paciente sem contexto completo
- Garantir a precisão de informações sobre casos individuais complexos
- Tomar decisões clínicas autônomas

---

## Situações de Emergência

O modelo está configurado para identificar e orientar sobre emergências. Caso o usuário descreva qualquer um dos cenários abaixo, o modelo **deve** recomendar atendimento imediato:

| Sintoma / Situação | Ação recomendada |
|---|---|
| Sangramento intenso na gestação | Emergência — SAMU 192 ou pronto-socorro |
| Dor abdominal intensa | Emergência — pronto-socorro |
| Febre acima de 38°C persistente | Avaliação médica urgente |
| Convulsões | Emergência — SAMU 192 |
| Falta de ar | Emergência — SAMU 192 |
| Dor no peito | Emergência — SAMU 192 |
| Desmaio | Emergência — SAMU 192 |
| Ideação suicida / autoagressão | CVV 188 + pronto-socorro psiquiátrico |
| Pressão alta + dor de cabeça + visão turva (gestante) | Emergência obstétrica imediata |
| Redução dos movimentos fetais | Maternidade imediatamente |
| Sangramento pós-parto intenso | Emergência — SAMU 192 |
| Violência com risco imediato | 190 (Polícia) + 180 (Central da Mulher) |

---

## Serviços de Apoio (Brasil)

| Serviço | Número | Disponibilidade |
|---|---|---|
| **SAMU** | 192 | 24h |
| **Bombeiros** | 193 | 24h |
| **Polícia** | 190 | 24h |
| **Central de Atendimento à Mulher** | 180 | 24h — gratuito e sigiloso |
| **CVV (Suicídio e Crise)** | 188 | 24h — gratuito |
| **Disque Denúncia (violência)** | 100 | 24h |

---

## Público-Alvo Recomendado

### Indicado para:
- Pacientes buscando educação em saúde e orientação inicial
- Profissionais de saúde como ferramenta de suporte informacional
- Sistemas hospitalares para apoio à triagem inicial
- Aplicativos de saúde feminina como componente informativo

### Não indicado para:
- Diagnóstico médico autônomo
- Prescrição eletrônica
- Substituição de prontuário eletrônico
- Terapia psicológica ou psiquiátrica
- Casos de emergência sem orientação para serviços de saúde

---

## Responsabilidade do Operador

Qualquer sistema que integre este modelo deve:

1. **Exibir aviso de isenção de responsabilidade** em todas as interações, informando que as respostas não substituem avaliação médica.
2. **Implementar filtro de saída** para detectar e bloquear respostas com prescrição direta.
3. **Registrar logs auditáveis** de todas as interações para revisão médica.
4. **Disponibilizar botão de emergência** ou atalho para serviços de saúde quando detectados sinais de risco.
5. **Obter validação periódica** de profissional de saúde qualificado, especialmente após atualizações do modelo.
6. **Cumprir a LGPD** (Lei Geral de Proteção de Dados) no armazenamento e tratamento de dados de saúde.
7. **Não usar como único canal** em situações de crise clínica.

---

## Limitações Conhecidas

### Limitações Técnicas
1. **Alucinação**: O modelo pode ocasionalmente gerar informações imprecisas ou fictícias com aparência de veracidade. Toda informação clínica deve ser verificada por profissional habilitado.
2. **Idioma**: Otimizado para português brasileiro. Pode ter desempenho reduzido em outros idiomas ou dialetos regionais muito específicos.
3. **Atualização de protocolos**: Baseado em dados de treinamento com data de corte. Protocolos médicos publicados após essa data não são incorporados automaticamente.
4. **Contexto limitado**: O modelo não tem acesso ao histórico médico completo do paciente. Respostas são baseadas apenas no que é descrito na conversa atual.
5. **Sem visão computacional**: Não interpreta imagens de exames, ultrassonografias, mamografias ou fotos de lesões.
6. **Sem integração com prontuário**: Não acessa dados do paciente em sistemas hospitalares.

### Limitações Clínicas
7. **Dosagem de medicamentos**: Nunca deve ser usada para cálculo ou prescrição de doses.
8. **Doenças raras**: O treinamento pode ser insuficiente para condições ginecológicas ou obstétricas raras.
9. **Comorbidades complexas**: Em pacientes com múltiplas condições simultâneas, a orientação pode ser insuficiente.
10. **Emergências obstétricas**: Mesmo quando identifica a emergência corretamente, não substitui monitorização fetal eletrônica, exames laboratoriais ou avaliação clínica presencial.
11. **Diversidade étnica e genética**: Pode apresentar lacunas para condições com maior prevalência em grupos étnicos específicos pouco representados no dataset.
12. **Saúde mental grave**: Não serve como psicoterapia, diagnóstico psiquiátrico ou manejo de crise suicida — apenas orientação para busca de ajuda.

### Limitações de Dados
13. **Dataset de violência doméstica pequeno**: Apenas 7 exemplos de alta qualidade foram utilizados. A capacidade do modelo neste domínio deve ser testada extensivamente antes do uso.
14. **Preponderância do inglês nos dados**: Parte do dataset `women-health-mini` está em inglês. Respostas em português foram priorizadas, mas pode haver degradação em tópicos não cobertos pelos outros datasets.

---

## Recomendações para Atualização Contínua

### Frequência de revisão
- **A cada 6 meses**: Revisar e incorporar novas diretrizes do Ministério da Saúde, FEBRASGO, CFM, SBP e OMS.
- **A cada novo major release**: Reavaliar o modelo-base para garantir uso da versão mais capaz e segura.
- **Após eventos clínicos**: Se casos de resposta inadequada forem reportados, investigar e corrigir imediatamente.

### Fontes de referência para atualização
- FEBRASGO (Federação Brasileira de Ginecologia e Obstetrícia)
- MS — Ministério da Saúde: Cadernos de Atenção Básica, protocolos de pré-natal
- OMS — Organização Mundial da Saúde: guias de saúde reprodutiva
- NICE (UK) e ACOG (EUA): diretrizes internacionais
- Publicações em periódicos indexados (PubMed, LILACS, SciELO)

### Processo recomendado de atualização
1. Coletar novos exemplos de QA clínica validados por especialistas
2. Executar `01_audit_data.py` nos novos dados
3. Executar `02_preprocess_data.py` — os novos dados serão mesclados aos existentes
4. Retornar `03_finetune.py` com fine-tuning adicional (continual learning)
5. Executar `04_evaluate.py --simulate` primeiro e depois com modelo real
6. Obter aprovação de especialista clínico antes de publicar nova versão

---

## Validação antes do uso clínico

O modelo NÃO deve ser implantado em ambiente clínico real sem:

- [ ] Avaliação por pelo menos um ginecologista/obstetra
- [ ] Revisão por especialista em violência doméstica (serviço social ou psicologia clínica)
- [ ] Taxa de reconhecimento de emergências >= 95% nos testes obrigatórios
- [ ] Aprovação pelo comitê de ética local (se aplicável ao contexto de uso)
- [ ] Implementação de mecanismo de feedback e reporte de erros
- [ ] Treinamento dos operadores sobre limitações do modelo
- [ ] Documentação de isenção de responsabilidade visível ao usuário final

---

## Contato e Reporte de Problemas

Caso identifique uma resposta inadequada, insegura ou clinicamente incorreta:

1. Registre o ID da conversa e o conteúdo completo
2. Anote o comportamento esperado e o observado
3. Encaminhe ao time responsável pela manutenção do modelo
4. Se o erro representar risco clínico imediato, suspenda o uso até correção

---

*Este documento deve ser distribuído junto com o modelo e revisado sempre que houver atualização.*
